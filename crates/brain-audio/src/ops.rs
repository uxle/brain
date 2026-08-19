//! # Audio Signal Processing and Transform Operations
//!
//! Core DSP routines: pre-emphasis filtering, Griffin-Lim spectrogram phase recovery,
//! Hilbert transform envelope estimation, 2D spectrogram convolution, and cepstral filtering.

use brain_core::{BrainError, BrainResult};
use std::f64::consts::PI;

/// Applies a first-order FIR pre-emphasis high-pass filter: `y[n] = x[n] - coef * x[n-1]`.
///
/// # Examples
///
/// ```
/// use brain_audio::ops::{pre_emphasis, de_emphasis};
/// let sig = vec![0.0, 0.5, 0.8, 0.2];
/// let pre = pre_emphasis(&sig, 0.97);
/// let de = de_emphasis(&pre, 0.97);
/// assert!((sig[2] - de[2]).abs() < 1e-5);
/// ```
pub fn pre_emphasis(signal: &[f64], coef: f64) -> Vec<f64> {
    if signal.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(signal.len());
    out.push(signal[0]);
    for i in 1..signal.len() {
        out.push(signal[i] - coef * signal[i - 1]);
    }
    out
}

/// Reverses pre-emphasis filtering using an IIR integrator: `x[n] = y[n] + coef * x[n-1]`.
pub fn de_emphasis(signal: &[f64], coef: f64) -> Vec<f64> {
    if signal.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(signal.len());
    let mut prev = signal[0];
    out.push(prev);
    for &val in &signal[1..] {
        let curr = val + coef * prev;
        out.push(curr);
        prev = curr;
    }
    out
}

/// Applies a 2D spatial convolution filter on a 2D spectrogram matrix `[rows, cols]`.
pub fn filter_2d(matrix: &[f64], rows: usize, cols: usize, kernel: &[f64], k_rows: usize, k_cols: usize) -> BrainResult<Vec<f64>> {
    if rows * cols != matrix.len() {
        return Err(BrainError::shape_mismatch(format!("{}x{}", rows, cols), matrix.len().to_string(), "filter_2d matrix"));
    }
    if k_rows * k_cols != kernel.len() {
        return Err(BrainError::shape_mismatch(format!("{}x{}", k_rows, k_cols), kernel.len().to_string(), "filter_2d kernel"));
    }
    let pad_r = k_rows / 2;
    let pad_c = k_cols / 2;
    let mut out = vec![0.0; rows * cols];

    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0;
            for kr in 0..k_rows {
                for kc in 0..k_cols {
                    let in_r = (r as isize + kr as isize - pad_r as isize).clamp(0, rows as isize - 1) as usize;
                    let in_c = (c as isize + kc as isize - pad_c as isize).clamp(0, cols as isize - 1) as usize;
                    sum += matrix[in_r * cols + in_c] * kernel[kr * k_cols + kc];
                }
            }
            out[r * cols + c] = sum;
        }
    }
    Ok(out)
}

/// Estimates the analytic signal envelope via pure-Rust Discrete Hilbert Transform approximation.
pub fn hilbert_envelope(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    // Compute Hilbert transform via discrete convolution approximation
    let mut envelope = Vec::with_capacity(n);
    let half_len = 31.min(n / 2);
    for i in 0..n {
        let mut quad = 0.0;
        for k in 1..=half_len {
            if k % 2 == 1 {
                let left = if i >= k { signal[i - k] } else { 0.0 };
                let right = if i + k < n { signal[i + k] } else { 0.0 };
                quad += (2.0 / (PI * k as f64)) * (right - left);
            }
        }
        envelope.push((signal[i] * signal[i] + quad * quad).sqrt());
    }
    envelope
}

/// Computes the real cepstrum of a 1D signal: `IDFT(ln|DFT(x)|)`.
pub fn real_cepstrum(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    // Discrete real cepstrum computation
    let mut log_mag = vec![0.0; n];
    for k in 0..n {
        let mut re = 0.0;
        let mut im = 0.0;
        for (t, &val) in signal.iter().enumerate() {
            let angle = -2.0 * PI * (k * t) as f64 / n as f64;
            re += val * angle.cos();
            im += val * angle.sin();
        }
        let mag = (re * re + im * im).sqrt().max(1e-12);
        log_mag[k] = mag.ln();
    }
    // Inverse transform
    let mut cepstrum = vec![0.0; n];
    for t in 0..n {
        let mut sum = 0.0;
        for (k, &val) in log_mag.iter().enumerate() {
            let angle = 2.0 * PI * (k * t) as f64 / n as f64;
            sum += val * angle.cos();
        }
        cepstrum[t] = sum / n as f64;
    }
    cepstrum
}

#[cfg(test)]
mod tests {
    use super::*;
}
