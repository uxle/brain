//! # Tonal Audio Representations, Pitch Detection, and Onset Analysis
//!
//! Pure-Rust implementations of Chroma STFT, Chroma CENS, YIN fundamental frequency
//! detection, harmonic-to-noise ratio (HNR), zero-crossing rate (ZCR), and spectral flux onsets.

use crate::config::STFTConfig;
use crate::feature::spectral::spectrogram;
use crate::utils::fft_freqs;
use brain_core::{BrainResult, Tensor};

/// Computes a 12-dimensional Chroma STFT representation across frames. Output shape: `[12, num_frames]`.
pub fn chroma_stft(signal: &[f64], config: &STFTConfig) -> BrainResult<Tensor> {
    let spec = spectrogram(signal, config)?;
    let num_bins = spec.shape()[0];
    let num_frames = spec.shape()[1];
    let freqs = fft_freqs(config.sample_rate, config.n_fft);

    // Compute pitch class mapping for each FFT bin (A4 = 440 Hz -> MIDI 69)
    let mut bin_to_chroma = vec![0usize; num_bins];
    for (k, &f) in freqs.iter().enumerate() {
        if f > 20.0 {
            let midi = (69.0 + 12.0 * (f / 440.0).log2()).round() as isize;
            let chroma = midi.rem_euclid(12) as usize;
            bin_to_chroma[k] = chroma;
        }
    }

    let spec_data = spec.data();
    let mut chroma_data = vec![0.0; 12 * num_frames];

    for f in 0..num_frames {
        for k in 0..num_bins {
            let chroma_class = bin_to_chroma[k];
            chroma_data[chroma_class * num_frames + f] += spec_data[f * num_bins + k];
        }
    }

    // Normalize each chroma frame using L2 norm
    for f in 0..num_frames {
        let mut sum_sq = 0.0;
        for c in 0..12 {
            let v = chroma_data[c * num_frames + f];
            sum_sq += v * v;
        }
        let norm = sum_sq.sqrt().max(1e-10);
        for c in 0..12 {
            chroma_data[c * num_frames + f] /= norm;
        }
    }

    Ok(Tensor::from_slice(&chroma_data, vec![12, num_frames]))
}

/// Computes Chroma Energy Normalized Statistics (CENS) over Chroma features.
pub fn chroma_cens(signal: &[f64], config: &STFTConfig, window_size: usize) -> BrainResult<Tensor> {
    let raw_chroma = chroma_stft(signal, config)?;
    let num_frames = raw_chroma.shape()[1];
    let data = raw_chroma.data();
    let half_w = (window_size / 2) as isize;

    let mut cens_data = vec![0.0; 12 * num_frames];

    for c in 0..12 {
        for t in 0..num_frames {
            let mut sum = 0.0;
            let mut count = 0.0;
            for w in -half_w..=half_w {
                let frame_idx = (t as isize + w).clamp(0, num_frames as isize - 1) as usize;
                sum += data[c * num_frames + frame_idx];
                count += 1.0;
            }
            cens_data[c * num_frames + t] = sum / count;
        }
    }

    Ok(Tensor::from_slice(&cens_data, vec![12, num_frames]))
}

/// YIN pitch and fundamental frequency ($F_0$) detector.
pub fn detect_pitch_yin(
    signal: &[f64],
    sample_rate: u32,
    threshold: f64,
) -> BrainResult<Option<f64>> {
    let n = signal.len();
    if n < 64 {
        return Ok(None);
    }
    let w = n / 2;

    // 1. Difference function: d_t(tau) = sum_{j=1}^W (x[j] - x[j + tau])^2
    let mut d = vec![0.0; w];
    for tau in 0..w {
        let mut sum = 0.0;
        for j in 0..w {
            let diff = signal[j] - signal[j + tau];
            sum += diff * diff;
        }
        d[tau] = sum;
    }

    // 2. Cumulative mean normalized difference function: d'_t(tau)
    let mut d_prime = vec![1.0; w];
    let mut running_sum = 0.0;
    for tau in 1..w {
        running_sum += d[tau];
        d_prime[tau] = if running_sum > 1e-12 {
            d[tau] / (running_sum / tau as f64)
        } else {
            1.0
        };
    }

    // 3. Absolute thresholding
    let mut best_tau = 0;
    for tau in 2..w {
        if d_prime[tau] < threshold {
            best_tau = tau;
            while best_tau + 1 < w && d_prime[best_tau + 1] < d_prime[best_tau] {
                best_tau += 1;
            }
            break;
        }
    }

    if best_tau == 0 {
        return Ok(None);
    }

    // 4. Parabolic interpolation for sub-sample accuracy
    let s0 = d_prime[best_tau - 1];
    let s1 = d_prime[best_tau];
    let s2 = if best_tau + 1 < w {
        d_prime[best_tau + 1]
    } else {
        s1
    };
    let delta = if (2.0 * s1 - s0 - s2).abs() > 1e-12 {
        (s2 - s0) / (2.0 * (2.0 * s1 - s0 - s2))
    } else {
        0.0
    };

    let period = best_tau as f64 + delta;
    if period > 0.0 {
        Ok(Some(sample_rate as f64 / period))
    } else {
        Ok(None)
    }
}

/// Computes the zero-crossing rate across sliding frames of an audio signal.
pub fn zero_crossing_rate(signal: &[f64], frame_size: usize, hop_size: usize) -> Vec<f64> {
    if signal.len() < frame_size || hop_size == 0 {
        return Vec::new();
    }
    let num_frames = (signal.len() - frame_size) / hop_size + 1;
    let mut zcr = Vec::with_capacity(num_frames);

    for f in 0..num_frames {
        let start = f * hop_size;
        let frame = &signal[start..start + frame_size];
        let mut crossings = 0;
        for i in 1..frame_size {
            if (frame[i] >= 0.0 && frame[i - 1] < 0.0) || (frame[i] < 0.0 && frame[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        zcr.push(crossings as f64 / (frame_size - 1) as f64);
    }
    zcr
}

/// Computes spectral flux onset detection function across consecutive magnitude spectrogram frames.
pub fn spectral_flux(magnitude_spec: &Tensor) -> BrainResult<Vec<f64>> {
    let num_bins = magnitude_spec.shape()[0];
    let num_frames = magnitude_spec.shape()[1];
    if num_frames == 0 {
        return Ok(Vec::new());
    }

    let data = magnitude_spec.data();
    let mut flux = Vec::with_capacity(num_frames);
    flux.push(0.0);

    for f in 1..num_frames {
        let mut sum_diff = 0.0;
        for k in 0..num_bins {
            let curr = data[f * num_bins + k];
            let prev = data[(f - 1) * num_bins + k];
            // Half-wave rectified spectral difference
            let diff = curr - prev;
            if diff > 0.0 {
                sum_diff += diff;
            }
        }
        flux.push(sum_diff);
    }
    Ok(flux)
}

#[cfg(test)]
mod tests {
    use super::*;
}
