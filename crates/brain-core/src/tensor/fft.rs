//! Pure-Rust Fast Fourier Transforms (FFT, IFFT, RFFT, IRFFT) and window functions.
//!
//! This module provides Cooley-Tukey Radix-2 discrete Fourier transforms and spectral windowing functions.

use crate::tensor::Tensor;

/// In-place Cooley-Tukey Radix-2 FFT/IFFT algorithm.
pub fn fft_radix2(real: &mut [f64], imag: &mut [f64], inverse: bool) {
    let n = real.len();
    assert_eq!(n, imag.len());
    assert!(n.is_power_of_two(), "FFT size must be a power of two");

    // Bit reversal permutation
    let mut j = 0;
    for i in 0..n {
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
        let mut bit = n >> 1;
        while bit > 0 && (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
    }

    // Cooley-Tukey butterfly computations
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_sign = if inverse { 1.0 } else { -1.0 };
        let angle = angle_sign * 2.0 * std::f64::consts::PI / (len as f64);
        let w_step_re = angle.cos();
        let w_step_im = angle.sin();

        let mut i = 0;
        while i < n {
            let mut w_re = 1.0;
            let mut w_im = 0.0;
            for k in 0..half {
                let u_re = real[i + k];
                let u_im = imag[i + k];
                let v_re = real[i + k + half] * w_re - imag[i + k + half] * w_im;
                let v_im = real[i + k + half] * w_im + imag[i + k + half] * w_re;

                real[i + k] = u_re + v_re;
                imag[i + k] = u_im + v_im;
                real[i + k + half] = u_re - v_re;
                imag[i + k + half] = u_im - v_im;

                let next_w_re = w_re * w_step_re - w_im * w_step_im;
                let next_w_im = w_re * w_step_im + w_im * w_step_re;
                w_re = next_w_re;
                w_im = next_w_im;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / (n as f64);
        for i in 0..n {
            real[i] *= inv_n;
            imag[i] *= inv_n;
        }
    }
}

/// Computes 1D Real Fast Fourier Transform: real input -> (real_part, imag_part).
pub fn rfft(input: &Tensor) -> (Tensor, Tensor) {
    assert_eq!(input.ndim(), 1);
    let n = input.numel();
    let mut real = input.to_vec();
    let mut imag = vec![0.0; n];
    fft_radix2(&mut real, &mut imag, false);
    let half_n = n / 2 + 1;
    (
        Tensor::new(real[..half_n].to_vec(), vec![half_n]),
        Tensor::new(imag[..half_n].to_vec(), vec![half_n]),
    )
}

/// Generates a Hann window of length `n`.
pub fn hann_window(n: usize) -> Tensor {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let val = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * (i as f64) / (n as f64 - 1.0)).cos());
        data.push(val);
    }
    Tensor::new(data, vec![n])
}

/// In-place Discrete Fourier Transform for arbitrary lengths (DFT / IDFT).
/// Automatically uses Cooley-Tukey Radix-2 FFT when `n` is a power of two.
pub fn dft(real: &mut [f64], imag: &mut [f64], inverse: bool) {
    let n = real.len();
    assert_eq!(n, imag.len());
    if n == 0 {
        return;
    }
    if n.is_power_of_two() {
        fft_radix2(real, imag, inverse);
        return;
    }

    let in_real = real.to_vec();
    let in_imag = imag.to_vec();
    let angle_sign = if inverse { 1.0 } else { -1.0 };
    let base_angle = angle_sign * 2.0 * std::f64::consts::PI / (n as f64);

    for k in 0..n {
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        for t in 0..n {
            let angle = base_angle * ((k * t) as f64);
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            sum_re += in_real[t] * cos_a - in_imag[t] * sin_a;
            sum_im += in_real[t] * sin_a + in_imag[t] * cos_a;
        }
        if inverse {
            real[k] = sum_re / (n as f64);
            imag[k] = sum_im / (n as f64);
        } else {
            real[k] = sum_re;
            imag[k] = sum_im;
        }
    }
}

/// Generates a Hamming window of length `n`.
pub fn hamming_window(n: usize) -> Tensor {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let val = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * (i as f64) / (n as f64 - 1.0)).cos();
        data.push(val);
    }
    Tensor::new(data, vec![n])
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_radix2_power_of_two_roundtrip() {
        for &size in &[4, 8, 16, 64, 128] {
            let orig_real: Vec<f64> = (0..size).map(|i| (i as f64 * 0.3).sin()).collect();
            let mut real = orig_real.clone();
            let mut imag = vec![0.0; size];

            fft_radix2(&mut real, &mut imag, false);
            fft_radix2(&mut real, &mut imag, true);

            for i in 0..size {
                assert!(
                    (real[i] - orig_real[i]).abs() < 1e-6,
                    "Power of two FFT roundtrip failed at size {}, idx {}",
                    size,
                    i
                );
                assert!(imag[i].abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_dft_non_power_of_two_roundtrip() {
        for &size in &[3, 5, 7, 13, 17, 25, 50] {
            let orig_real: Vec<f64> = (0..size).map(|i| (i as f64 * 0.5).cos()).collect();
            let mut real = orig_real.clone();
            let mut imag = vec![0.0; size];

            dft(&mut real, &mut imag, false);
            dft(&mut real, &mut imag, true);

            for i in 0..size {
                assert!(
                    (real[i] - orig_real[i]).abs() < 1e-6,
                    "Non-power of two DFT roundtrip failed at size {}, idx {}",
                    size,
                    i
                );
                assert!(imag[i].abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_rfft_and_windows() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let (re, im) = rfft(&t);
        assert_eq!(re.shape(), &[3]);
        assert_eq!(im.shape(), &[3]);
        assert_eq!(re.data()[0], 10.0); // DC component is sum of elements

        let hann = hann_window(16);
        assert_eq!(hann.shape(), &[16]);
        assert_eq!(hann.data()[0], 0.0);

        let hamm = hamming_window(16);
        assert_eq!(hamm.shape(), &[16]);
        assert!((hamm.data()[0] - 0.08).abs() < 1e-6);
    }
}
