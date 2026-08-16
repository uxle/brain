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
        return Err(BrainError::invalid_value("spec_augment requires 2D [channels, frames] tensor"));
    }
    let n_freq = spectrogram.shape()[0];
    let n_time = spectrogram.shape()[1];

    let mut state = seed.wrapping_add(0xdeadbeef);

    // Apply frequency masks
    for _ in 0..num_freq_masks {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let f_len = (state >> 32) as usize % freq_mask_param.min(n_freq);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let f_start = (state >> 32) as usize % (n_freq - f_len).max(1);
        frequency_mask(spectrogram, f_start, f_len)?;
    }

    // Apply time masks
    for _ in 0..num_time_masks {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let t_len = (state >> 32) as usize % time_mask_param.min(n_time);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
pub fn spec_cutout(spectrogram: &mut Tensor, f_start: usize, f_len: usize, t_start: usize, t_len: usize) -> BrainResult<()> {
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
    let mixed: Vec<f64> = d1.iter().zip(d2.iter()).map(|(&x, &y)| a * x + b * y).collect();
    Ok(Tensor::from_slice(&mixed, spec1.shape().to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_aug_stress_001() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 1) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 1 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_002() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 2) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 2 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_003() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 3) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 3 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_004() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 4) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 4 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_005() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 5) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 5 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_006() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 6) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 6 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_007() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 7) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 7 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_008() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 8) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 8 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_009() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 9) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 9 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_010() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 10) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 10 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_011() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 11) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 11 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_012() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 12) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 12 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_013() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 13) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 13 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_014() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 14) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 14 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_015() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 15) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 15 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_016() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 16) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 16 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_017() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 17) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 17 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_018() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 18) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 18 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_019() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 19) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 19 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_020() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 20) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 20 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_021() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 21) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 21 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_022() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 22) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 22 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_023() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 23) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 23 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_024() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 24) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 24 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_025() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 25) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 25 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_026() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 26) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 26 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_027() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 27) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 27 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_028() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 28) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 28 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_029() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 29) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 29 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_030() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 30) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 30 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_031() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 31) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 31 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_032() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 32) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 32 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_033() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 33) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 33 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_034() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 34) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 34 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_035() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 35) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 35 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_036() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 36) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 36 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_037() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 37) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 37 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_038() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 38) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 38 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_039() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 39) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 39 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_040() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 40) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 40 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_041() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 41) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 41 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_042() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 42) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 42 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_043() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 43) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 43 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_044() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 44) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 44 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_045() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 45) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 45 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_046() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 46) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 46 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_047() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 47) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 47 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_048() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 48) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 48 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_049() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 49) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 49 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_050() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 50) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 50 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_051() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 51) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 51 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_052() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 52) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 52 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_053() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 53) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 53 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_054() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 54) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 54 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_055() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 55) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 55 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_056() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 56) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 56 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_057() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 57) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 57 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_058() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 58) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 58 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_059() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 59) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 59 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_060() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 60) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 60 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_061() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 61) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 61 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_062() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 62) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 62 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_063() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 63) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 63 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_064() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 64) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 64 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_065() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 65) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 65 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_066() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 66) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 66 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_067() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 67) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 67 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_068() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 68) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 68 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_069() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 69) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 69 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_070() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 70) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 70 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_071() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 71) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 71 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_072() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 72) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 72 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_073() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 73) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 73 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_074() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 74) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 74 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_075() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 75) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 75 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_076() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 76) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 76 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_077() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 77) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 77 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_078() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 78) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 78 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_079() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 79) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 79 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_080() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 80) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 80 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_081() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 81) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 81 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_082() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 82) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 82 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_083() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 83) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 83 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_084() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 84) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 84 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_085() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 85) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 85 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_086() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 86) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 86 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_087() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 87) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 87 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_088() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 88) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 88 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_089() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 89) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 89 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_090() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 90) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 90 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_091() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 91) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 91 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_092() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 92) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 92 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_093() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 93) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 93 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_094() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 94) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 94 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_095() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 95) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 95 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_096() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 96) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 96 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_097() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 97) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 97 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_098() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 98) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 98 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_099() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 99) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 99 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_100() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 100) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 100 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_101() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 101) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 101 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_102() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 102) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 102 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_103() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 103) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 103 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_104() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 104) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 104 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_105() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 105) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 105 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_106() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 106) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 106 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_107() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 107) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 107 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_108() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 108) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 108 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_109() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 109) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 109 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_110() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 110) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 110 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_111() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 111) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 111 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_112() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 112) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 112 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_113() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 113) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 113 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_114() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 114) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 114 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_115() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 115) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 115 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_116() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 116) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 116 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_117() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 117) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 117 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_118() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 118) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 118 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_119() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 119) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 119 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_120() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 120) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 120 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_121() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 121) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 121 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_122() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 122) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 122 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_123() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 123) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 123 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_124() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 124) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 124 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_125() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 125) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 125 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_126() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 126) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 126 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_127() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 127) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 127 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_128() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 128) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 128 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_129() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 129) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 129 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_130() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 130) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 130 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_131() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 131) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 131 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_132() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 132) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 132 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_133() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 133) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 133 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_134() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 134) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 134 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_135() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 135) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 135 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_136() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 136) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 136 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_137() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 137) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 137 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_138() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 138) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 138 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_139() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 139) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 139 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_140() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 140) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 140 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_141() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 141) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 141 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_142() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 142) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 142 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_143() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 143) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 143 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_144() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 144) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 144 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_145() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 145) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 145 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_146() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 146) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 146 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_147() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 147) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 147 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_148() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 148) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 148 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_149() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 149) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 149 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_150() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 150) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 150 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_151() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 151) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 151 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_152() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 152) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 152 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_153() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 153) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 153 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_154() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 154) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 154 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_155() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 155) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 155 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_156() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 156) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 156 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_157() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 157) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 157 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_158() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 158) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 158 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_159() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 159) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 159 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_160() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 160) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 160 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_161() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 161) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 161 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_162() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 162) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 162 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_163() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 163) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 163 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_164() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 164) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 164 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_165() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 165) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 165 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_166() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 166) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 166 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_167() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 167) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 167 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_168() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 168) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 168 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_169() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 169) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 169 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_170() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 170) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 170 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_171() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 171) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 171 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_172() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 172) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 172 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_173() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 173) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 173 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_174() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 174) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 174 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_175() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 175) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 175 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_176() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 176) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 176 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_177() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 177) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 177 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_178() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 178) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 178 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_179() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 179) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 179 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_180() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 180) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 180 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_181() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 181) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 181 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_182() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 182) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 182 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_183() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 183) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 183 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_184() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 184) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 184 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_185() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 185) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 185 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_186() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 186) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 186 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_187() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 187) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 187 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_188() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 188) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 188 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_189() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 189) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 189 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_190() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 190) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 190 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_191() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 191) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 191 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_192() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 192) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 192 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_193() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 193) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 193 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_194() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 194) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 194 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_195() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 195) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 195 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_196() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 196) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 196 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_197() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 197) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 197 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_198() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 198) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 198 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_199() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 199) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 199 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_200() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 200) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 200 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_201() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 201) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 201 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_202() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 202) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 202 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_203() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 203) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 203 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_204() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 204) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 204 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_205() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 205) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 205 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_206() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 206) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 206 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_207() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 207) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 207 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_208() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 208) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 208 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_209() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 209) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 209 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_210() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 210) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 210 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_211() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 211) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 211 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_212() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 212) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 212 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_213() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 213) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 213 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_214() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 214) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 214 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_215() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 215) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 215 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_216() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 216) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 216 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_217() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 217) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 217 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_218() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 218) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 218 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_219() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 219) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 219 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_220() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 220) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 220 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_221() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 221) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 221 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_222() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 222) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 222 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_223() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 223) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 223 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_224() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 224) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 224 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_225() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 225) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 225 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_226() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 226) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 226 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_227() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 227) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 227 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_228() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 228) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 228 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_229() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 229) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 229 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_230() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 230) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 230 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }

    #[test]
    fn test_spectral_aug_stress_231() {
        let mut spec = Tensor::zeros(vec![80, 100]);
        for i in 0..80 * 100 {
            spec.data_mut()[i] = ((i + 231) as f64 * 0.01).cos();
        }
        spec_augment(&mut spec, 10, 2, 15, 2, 231 as u64).unwrap();
        spec_cutout(&mut spec, 5, 10, 10, 20).unwrap();
        
        let spec2 = spec.clone();
        let mixed = spec_mixup(&spec, &spec2, 0.5).unwrap();
        assert_eq!(mixed.shape(), spec.shape());
    }
}
