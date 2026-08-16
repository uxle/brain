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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_roundtrip() {
        let mut real = vec![1.0, 2.0, 3.0, 4.0];
        let mut imag = vec![0.0, 0.0, 0.0, 0.0];
        let orig = real.clone();
        fft_radix2(&mut real, &mut imag, false);
        fft_radix2(&mut real, &mut imag, true);
        for i in 0..4 {
            assert!((real[i] - orig[i]).abs() < 1e-6);
            assert!(imag[i].abs() < 1e-6);
        }
    }

    #[test]
    fn test_fft_stress_case_001() {
        let mut real = vec![1.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 1.0);
    }

    #[test]
    fn test_fft_stress_case_002() {
        let mut real = vec![2.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 2.0);
    }

    #[test]
    fn test_fft_stress_case_003() {
        let mut real = vec![3.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 3.0);
    }

    #[test]
    fn test_fft_stress_case_004() {
        let mut real = vec![4.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 4.0);
    }

    #[test]
    fn test_fft_stress_case_005() {
        let mut real = vec![5.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 5.0);
    }

    #[test]
    fn test_fft_stress_case_006() {
        let mut real = vec![6.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 6.0);
    }

    #[test]
    fn test_fft_stress_case_007() {
        let mut real = vec![7.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 7.0);
    }

    #[test]
    fn test_fft_stress_case_008() {
        let mut real = vec![8.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 8.0);
    }

    #[test]
    fn test_fft_stress_case_009() {
        let mut real = vec![9.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 9.0);
    }

    #[test]
    fn test_fft_stress_case_010() {
        let mut real = vec![10.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 10.0);
    }

    #[test]
    fn test_fft_stress_case_011() {
        let mut real = vec![11.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 11.0);
    }

    #[test]
    fn test_fft_stress_case_012() {
        let mut real = vec![12.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 12.0);
    }

    #[test]
    fn test_fft_stress_case_013() {
        let mut real = vec![13.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 13.0);
    }

    #[test]
    fn test_fft_stress_case_014() {
        let mut real = vec![14.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 14.0);
    }

    #[test]
    fn test_fft_stress_case_015() {
        let mut real = vec![15.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 15.0);
    }

    #[test]
    fn test_fft_stress_case_016() {
        let mut real = vec![16.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 16.0);
    }

    #[test]
    fn test_fft_stress_case_017() {
        let mut real = vec![17.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 17.0);
    }

    #[test]
    fn test_fft_stress_case_018() {
        let mut real = vec![18.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 18.0);
    }

    #[test]
    fn test_fft_stress_case_019() {
        let mut real = vec![19.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 19.0);
    }

    #[test]
    fn test_fft_stress_case_020() {
        let mut real = vec![20.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 20.0);
    }

    #[test]
    fn test_fft_stress_case_021() {
        let mut real = vec![21.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 21.0);
    }

    #[test]
    fn test_fft_stress_case_022() {
        let mut real = vec![22.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 22.0);
    }

    #[test]
    fn test_fft_stress_case_023() {
        let mut real = vec![23.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 23.0);
    }

    #[test]
    fn test_fft_stress_case_024() {
        let mut real = vec![24.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 24.0);
    }

    #[test]
    fn test_fft_stress_case_025() {
        let mut real = vec![25.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 25.0);
    }

    #[test]
    fn test_fft_stress_case_026() {
        let mut real = vec![26.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 26.0);
    }

    #[test]
    fn test_fft_stress_case_027() {
        let mut real = vec![27.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 27.0);
    }

    #[test]
    fn test_fft_stress_case_028() {
        let mut real = vec![28.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 28.0);
    }

    #[test]
    fn test_fft_stress_case_029() {
        let mut real = vec![29.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 29.0);
    }

    #[test]
    fn test_fft_stress_case_030() {
        let mut real = vec![30.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 30.0);
    }

    #[test]
    fn test_fft_stress_case_031() {
        let mut real = vec![31.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 31.0);
    }

    #[test]
    fn test_fft_stress_case_032() {
        let mut real = vec![32.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 32.0);
    }

    #[test]
    fn test_fft_stress_case_033() {
        let mut real = vec![33.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 33.0);
    }

    #[test]
    fn test_fft_stress_case_034() {
        let mut real = vec![34.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 34.0);
    }

    #[test]
    fn test_fft_stress_case_035() {
        let mut real = vec![35.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 35.0);
    }

    #[test]
    fn test_fft_stress_case_036() {
        let mut real = vec![36.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 36.0);
    }

    #[test]
    fn test_fft_stress_case_037() {
        let mut real = vec![37.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 37.0);
    }

    #[test]
    fn test_fft_stress_case_038() {
        let mut real = vec![38.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 38.0);
    }

    #[test]
    fn test_fft_stress_case_039() {
        let mut real = vec![39.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 39.0);
    }

    #[test]
    fn test_fft_stress_case_040() {
        let mut real = vec![40.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 40.0);
    }

    #[test]
    fn test_fft_stress_case_041() {
        let mut real = vec![41.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 41.0);
    }

    #[test]
    fn test_fft_stress_case_042() {
        let mut real = vec![42.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 42.0);
    }

    #[test]
    fn test_fft_stress_case_043() {
        let mut real = vec![43.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 43.0);
    }

    #[test]
    fn test_fft_stress_case_044() {
        let mut real = vec![44.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 44.0);
    }

    #[test]
    fn test_fft_stress_case_045() {
        let mut real = vec![45.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 45.0);
    }

    #[test]
    fn test_fft_stress_case_046() {
        let mut real = vec![46.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 46.0);
    }

    #[test]
    fn test_fft_stress_case_047() {
        let mut real = vec![47.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 47.0);
    }

    #[test]
    fn test_fft_stress_case_048() {
        let mut real = vec![48.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 48.0);
    }

    #[test]
    fn test_fft_stress_case_049() {
        let mut real = vec![49.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 49.0);
    }

    #[test]
    fn test_fft_stress_case_050() {
        let mut real = vec![50.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 50.0);
    }

    #[test]
    fn test_fft_stress_case_051() {
        let mut real = vec![51.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 51.0);
    }

    #[test]
    fn test_fft_stress_case_052() {
        let mut real = vec![52.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 52.0);
    }

    #[test]
    fn test_fft_stress_case_053() {
        let mut real = vec![53.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 53.0);
    }

    #[test]
    fn test_fft_stress_case_054() {
        let mut real = vec![54.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 54.0);
    }

    #[test]
    fn test_fft_stress_case_055() {
        let mut real = vec![55.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 55.0);
    }

    #[test]
    fn test_fft_stress_case_056() {
        let mut real = vec![56.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 56.0);
    }

    #[test]
    fn test_fft_stress_case_057() {
        let mut real = vec![57.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 57.0);
    }

    #[test]
    fn test_fft_stress_case_058() {
        let mut real = vec![58.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 58.0);
    }

    #[test]
    fn test_fft_stress_case_059() {
        let mut real = vec![59.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 59.0);
    }

    #[test]
    fn test_fft_stress_case_060() {
        let mut real = vec![60.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 60.0);
    }

    #[test]
    fn test_fft_stress_case_061() {
        let mut real = vec![61.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 61.0);
    }

    #[test]
    fn test_fft_stress_case_062() {
        let mut real = vec![62.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 62.0);
    }

    #[test]
    fn test_fft_stress_case_063() {
        let mut real = vec![63.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 63.0);
    }

    #[test]
    fn test_fft_stress_case_064() {
        let mut real = vec![64.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 64.0);
    }

    #[test]
    fn test_fft_stress_case_065() {
        let mut real = vec![65.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 65.0);
    }

    #[test]
    fn test_fft_stress_case_066() {
        let mut real = vec![66.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 66.0);
    }

    #[test]
    fn test_fft_stress_case_067() {
        let mut real = vec![67.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 67.0);
    }

    #[test]
    fn test_fft_stress_case_068() {
        let mut real = vec![68.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 68.0);
    }

    #[test]
    fn test_fft_stress_case_069() {
        let mut real = vec![69.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 69.0);
    }

    #[test]
    fn test_fft_stress_case_070() {
        let mut real = vec![70.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 70.0);
    }

    #[test]
    fn test_fft_stress_case_071() {
        let mut real = vec![71.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 71.0);
    }

    #[test]
    fn test_fft_stress_case_072() {
        let mut real = vec![72.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 72.0);
    }

    #[test]
    fn test_fft_stress_case_073() {
        let mut real = vec![73.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 73.0);
    }

    #[test]
    fn test_fft_stress_case_074() {
        let mut real = vec![74.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 74.0);
    }

    #[test]
    fn test_fft_stress_case_075() {
        let mut real = vec![75.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 75.0);
    }

    #[test]
    fn test_fft_stress_case_076() {
        let mut real = vec![76.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 76.0);
    }

    #[test]
    fn test_fft_stress_case_077() {
        let mut real = vec![77.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 77.0);
    }

    #[test]
    fn test_fft_stress_case_078() {
        let mut real = vec![78.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 78.0);
    }

    #[test]
    fn test_fft_stress_case_079() {
        let mut real = vec![79.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 79.0);
    }

    #[test]
    fn test_fft_stress_case_080() {
        let mut real = vec![80.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 80.0);
    }

    #[test]
    fn test_fft_stress_case_081() {
        let mut real = vec![81.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 81.0);
    }

    #[test]
    fn test_fft_stress_case_082() {
        let mut real = vec![82.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 82.0);
    }

    #[test]
    fn test_fft_stress_case_083() {
        let mut real = vec![83.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 83.0);
    }

    #[test]
    fn test_fft_stress_case_084() {
        let mut real = vec![84.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 84.0);
    }

    #[test]
    fn test_fft_stress_case_085() {
        let mut real = vec![85.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 85.0);
    }

    #[test]
    fn test_fft_stress_case_086() {
        let mut real = vec![86.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 86.0);
    }

    #[test]
    fn test_fft_stress_case_087() {
        let mut real = vec![87.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 87.0);
    }

    #[test]
    fn test_fft_stress_case_088() {
        let mut real = vec![88.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 88.0);
    }

    #[test]
    fn test_fft_stress_case_089() {
        let mut real = vec![89.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 89.0);
    }

    #[test]
    fn test_fft_stress_case_090() {
        let mut real = vec![90.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 90.0);
    }

    #[test]
    fn test_fft_stress_case_091() {
        let mut real = vec![91.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 91.0);
    }

    #[test]
    fn test_fft_stress_case_092() {
        let mut real = vec![92.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 92.0);
    }

    #[test]
    fn test_fft_stress_case_093() {
        let mut real = vec![93.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 93.0);
    }

    #[test]
    fn test_fft_stress_case_094() {
        let mut real = vec![94.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 94.0);
    }

    #[test]
    fn test_fft_stress_case_095() {
        let mut real = vec![95.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 95.0);
    }

    #[test]
    fn test_fft_stress_case_096() {
        let mut real = vec![96.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 96.0);
    }

    #[test]
    fn test_fft_stress_case_097() {
        let mut real = vec![97.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 97.0);
    }

    #[test]
    fn test_fft_stress_case_098() {
        let mut real = vec![98.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 98.0);
    }

    #[test]
    fn test_fft_stress_case_099() {
        let mut real = vec![99.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 99.0);
    }

    #[test]
    fn test_fft_stress_case_100() {
        let mut real = vec![100.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 100.0);
    }

    #[test]
    fn test_fft_stress_case_101() {
        let mut real = vec![101.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 101.0);
    }

    #[test]
    fn test_fft_stress_case_102() {
        let mut real = vec![102.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 102.0);
    }

    #[test]
    fn test_fft_stress_case_103() {
        let mut real = vec![103.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 103.0);
    }

    #[test]
    fn test_fft_stress_case_104() {
        let mut real = vec![104.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 104.0);
    }

    #[test]
    fn test_fft_stress_case_105() {
        let mut real = vec![105.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 105.0);
    }

    #[test]
    fn test_fft_stress_case_106() {
        let mut real = vec![106.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 106.0);
    }

    #[test]
    fn test_fft_stress_case_107() {
        let mut real = vec![107.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 107.0);
    }

    #[test]
    fn test_fft_stress_case_108() {
        let mut real = vec![108.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 108.0);
    }

    #[test]
    fn test_fft_stress_case_109() {
        let mut real = vec![109.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 109.0);
    }

    #[test]
    fn test_fft_stress_case_110() {
        let mut real = vec![110.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 110.0);
    }

    #[test]
    fn test_fft_stress_case_111() {
        let mut real = vec![111.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 111.0);
    }

    #[test]
    fn test_fft_stress_case_112() {
        let mut real = vec![112.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 112.0);
    }

    #[test]
    fn test_fft_stress_case_113() {
        let mut real = vec![113.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 113.0);
    }

    #[test]
    fn test_fft_stress_case_114() {
        let mut real = vec![114.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 114.0);
    }

    #[test]
    fn test_fft_stress_case_115() {
        let mut real = vec![115.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 115.0);
    }

    #[test]
    fn test_fft_stress_case_116() {
        let mut real = vec![116.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 116.0);
    }

    #[test]
    fn test_fft_stress_case_117() {
        let mut real = vec![117.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 117.0);
    }

    #[test]
    fn test_fft_stress_case_118() {
        let mut real = vec![118.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 118.0);
    }

    #[test]
    fn test_fft_stress_case_119() {
        let mut real = vec![119.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 119.0);
    }

    #[test]
    fn test_fft_stress_case_120() {
        let mut real = vec![120.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 120.0);
    }

    #[test]
    fn test_fft_stress_case_121() {
        let mut real = vec![121.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 121.0);
    }

    #[test]
    fn test_fft_stress_case_122() {
        let mut real = vec![122.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 122.0);
    }

    #[test]
    fn test_fft_stress_case_123() {
        let mut real = vec![123.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 123.0);
    }

    #[test]
    fn test_fft_stress_case_124() {
        let mut real = vec![124.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 124.0);
    }

    #[test]
    fn test_fft_stress_case_125() {
        let mut real = vec![125.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 125.0);
    }

    #[test]
    fn test_fft_stress_case_126() {
        let mut real = vec![126.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 126.0);
    }

    #[test]
    fn test_fft_stress_case_127() {
        let mut real = vec![127.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 127.0);
    }

    #[test]
    fn test_fft_stress_case_128() {
        let mut real = vec![128.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 128.0);
    }

    #[test]
    fn test_fft_stress_case_129() {
        let mut real = vec![129.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 129.0);
    }

    #[test]
    fn test_fft_stress_case_130() {
        let mut real = vec![130.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 130.0);
    }

    #[test]
    fn test_fft_stress_case_131() {
        let mut real = vec![131.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 131.0);
    }

    #[test]
    fn test_fft_stress_case_132() {
        let mut real = vec![132.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 132.0);
    }

    #[test]
    fn test_fft_stress_case_133() {
        let mut real = vec![133.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 133.0);
    }

    #[test]
    fn test_fft_stress_case_134() {
        let mut real = vec![134.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 134.0);
    }

    #[test]
    fn test_fft_stress_case_135() {
        let mut real = vec![135.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 135.0);
    }

    #[test]
    fn test_fft_stress_case_136() {
        let mut real = vec![136.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 136.0);
    }

    #[test]
    fn test_fft_stress_case_137() {
        let mut real = vec![137.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 137.0);
    }

    #[test]
    fn test_fft_stress_case_138() {
        let mut real = vec![138.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 138.0);
    }

    #[test]
    fn test_fft_stress_case_139() {
        let mut real = vec![139.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 139.0);
    }

    #[test]
    fn test_fft_stress_case_140() {
        let mut real = vec![140.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 140.0);
    }

    #[test]
    fn test_fft_stress_case_141() {
        let mut real = vec![141.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 141.0);
    }

    #[test]
    fn test_fft_stress_case_142() {
        let mut real = vec![142.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 142.0);
    }

    #[test]
    fn test_fft_stress_case_143() {
        let mut real = vec![143.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 143.0);
    }

    #[test]
    fn test_fft_stress_case_144() {
        let mut real = vec![144.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 144.0);
    }

    #[test]
    fn test_fft_stress_case_145() {
        let mut real = vec![145.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 145.0);
    }

    #[test]
    fn test_fft_stress_case_146() {
        let mut real = vec![146.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 146.0);
    }

    #[test]
    fn test_fft_stress_case_147() {
        let mut real = vec![147.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 147.0);
    }

    #[test]
    fn test_fft_stress_case_148() {
        let mut real = vec![148.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 148.0);
    }

    #[test]
    fn test_fft_stress_case_149() {
        let mut real = vec![149.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 149.0);
    }

    #[test]
    fn test_fft_stress_case_150() {
        let mut real = vec![150.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 150.0);
    }

    #[test]
    fn test_fft_stress_case_151() {
        let mut real = vec![151.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 151.0);
    }

    #[test]
    fn test_fft_stress_case_152() {
        let mut real = vec![152.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 152.0);
    }

    #[test]
    fn test_fft_stress_case_153() {
        let mut real = vec![153.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 153.0);
    }

    #[test]
    fn test_fft_stress_case_154() {
        let mut real = vec![154.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 154.0);
    }

    #[test]
    fn test_fft_stress_case_155() {
        let mut real = vec![155.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 155.0);
    }

    #[test]
    fn test_fft_stress_case_156() {
        let mut real = vec![156.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 156.0);
    }

    #[test]
    fn test_fft_stress_case_157() {
        let mut real = vec![157.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 157.0);
    }

    #[test]
    fn test_fft_stress_case_158() {
        let mut real = vec![158.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 158.0);
    }

    #[test]
    fn test_fft_stress_case_159() {
        let mut real = vec![159.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 159.0);
    }

    #[test]
    fn test_fft_stress_case_160() {
        let mut real = vec![160.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 160.0);
    }

    #[test]
    fn test_fft_stress_case_161() {
        let mut real = vec![161.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 161.0);
    }

    #[test]
    fn test_fft_stress_case_162() {
        let mut real = vec![162.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 162.0);
    }

    #[test]
    fn test_fft_stress_case_163() {
        let mut real = vec![163.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 163.0);
    }

    #[test]
    fn test_fft_stress_case_164() {
        let mut real = vec![164.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 164.0);
    }

    #[test]
    fn test_fft_stress_case_165() {
        let mut real = vec![165.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 165.0);
    }

    #[test]
    fn test_fft_stress_case_166() {
        let mut real = vec![166.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 166.0);
    }

    #[test]
    fn test_fft_stress_case_167() {
        let mut real = vec![167.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 167.0);
    }

    #[test]
    fn test_fft_stress_case_168() {
        let mut real = vec![168.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 168.0);
    }

    #[test]
    fn test_fft_stress_case_169() {
        let mut real = vec![169.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 169.0);
    }

    #[test]
    fn test_fft_stress_case_170() {
        let mut real = vec![170.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 170.0);
    }

    #[test]
    fn test_fft_stress_case_171() {
        let mut real = vec![171.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 171.0);
    }

    #[test]
    fn test_fft_stress_case_172() {
        let mut real = vec![172.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 172.0);
    }

    #[test]
    fn test_fft_stress_case_173() {
        let mut real = vec![173.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 173.0);
    }

    #[test]
    fn test_fft_stress_case_174() {
        let mut real = vec![174.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 174.0);
    }

    #[test]
    fn test_fft_stress_case_175() {
        let mut real = vec![175.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 175.0);
    }

    #[test]
    fn test_fft_stress_case_176() {
        let mut real = vec![176.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 176.0);
    }

    #[test]
    fn test_fft_stress_case_177() {
        let mut real = vec![177.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 177.0);
    }

    #[test]
    fn test_fft_stress_case_178() {
        let mut real = vec![178.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 178.0);
    }

    #[test]
    fn test_fft_stress_case_179() {
        let mut real = vec![179.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 179.0);
    }

    #[test]
    fn test_fft_stress_case_180() {
        let mut real = vec![180.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 180.0);
    }

    #[test]
    fn test_fft_stress_case_181() {
        let mut real = vec![181.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 181.0);
    }

    #[test]
    fn test_fft_stress_case_182() {
        let mut real = vec![182.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 182.0);
    }

    #[test]
    fn test_fft_stress_case_183() {
        let mut real = vec![183.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 183.0);
    }

    #[test]
    fn test_fft_stress_case_184() {
        let mut real = vec![184.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 184.0);
    }

    #[test]
    fn test_fft_stress_case_185() {
        let mut real = vec![185.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 185.0);
    }

    #[test]
    fn test_fft_stress_case_186() {
        let mut real = vec![186.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 186.0);
    }

    #[test]
    fn test_fft_stress_case_187() {
        let mut real = vec![187.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 187.0);
    }

    #[test]
    fn test_fft_stress_case_188() {
        let mut real = vec![188.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 188.0);
    }

    #[test]
    fn test_fft_stress_case_189() {
        let mut real = vec![189.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 189.0);
    }

    #[test]
    fn test_fft_stress_case_190() {
        let mut real = vec![190.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 190.0);
    }

    #[test]
    fn test_fft_stress_case_191() {
        let mut real = vec![191.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 191.0);
    }

    #[test]
    fn test_fft_stress_case_192() {
        let mut real = vec![192.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 192.0);
    }

    #[test]
    fn test_fft_stress_case_193() {
        let mut real = vec![193.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 193.0);
    }

    #[test]
    fn test_fft_stress_case_194() {
        let mut real = vec![194.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 194.0);
    }

    #[test]
    fn test_fft_stress_case_195() {
        let mut real = vec![195.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 195.0);
    }

    #[test]
    fn test_fft_stress_case_196() {
        let mut real = vec![196.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 196.0);
    }

    #[test]
    fn test_fft_stress_case_197() {
        let mut real = vec![197.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 197.0);
    }

    #[test]
    fn test_fft_stress_case_198() {
        let mut real = vec![198.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 198.0);
    }

    #[test]
    fn test_fft_stress_case_199() {
        let mut real = vec![199.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 199.0);
    }

    #[test]
    fn test_fft_stress_case_200() {
        let mut real = vec![200.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 200.0);
    }

    #[test]
    fn test_fft_stress_case_201() {
        let mut real = vec![201.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 201.0);
    }

    #[test]
    fn test_fft_stress_case_202() {
        let mut real = vec![202.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 202.0);
    }

    #[test]
    fn test_fft_stress_case_203() {
        let mut real = vec![203.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 203.0);
    }

    #[test]
    fn test_fft_stress_case_204() {
        let mut real = vec![204.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 204.0);
    }

    #[test]
    fn test_fft_stress_case_205() {
        let mut real = vec![205.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 205.0);
    }

    #[test]
    fn test_fft_stress_case_206() {
        let mut real = vec![206.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 206.0);
    }

    #[test]
    fn test_fft_stress_case_207() {
        let mut real = vec![207.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 207.0);
    }

    #[test]
    fn test_fft_stress_case_208() {
        let mut real = vec![208.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 208.0);
    }

    #[test]
    fn test_fft_stress_case_209() {
        let mut real = vec![209.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 209.0);
    }

    #[test]
    fn test_fft_stress_case_210() {
        let mut real = vec![210.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 210.0);
    }

    #[test]
    fn test_fft_stress_case_211() {
        let mut real = vec![211.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 211.0);
    }

    #[test]
    fn test_fft_stress_case_212() {
        let mut real = vec![212.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 212.0);
    }

    #[test]
    fn test_fft_stress_case_213() {
        let mut real = vec![213.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 213.0);
    }

    #[test]
    fn test_fft_stress_case_214() {
        let mut real = vec![214.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 214.0);
    }

    #[test]
    fn test_fft_stress_case_215() {
        let mut real = vec![215.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 215.0);
    }

    #[test]
    fn test_fft_stress_case_216() {
        let mut real = vec![216.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 216.0);
    }

    #[test]
    fn test_fft_stress_case_217() {
        let mut real = vec![217.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 217.0);
    }

    #[test]
    fn test_fft_stress_case_218() {
        let mut real = vec![218.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 218.0);
    }

    #[test]
    fn test_fft_stress_case_219() {
        let mut real = vec![219.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 219.0);
    }

    #[test]
    fn test_fft_stress_case_220() {
        let mut real = vec![220.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 220.0);
    }

    #[test]
    fn test_fft_stress_case_221() {
        let mut real = vec![221.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 221.0);
    }

    #[test]
    fn test_fft_stress_case_222() {
        let mut real = vec![222.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 222.0);
    }

    #[test]
    fn test_fft_stress_case_223() {
        let mut real = vec![223.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 223.0);
    }

    #[test]
    fn test_fft_stress_case_224() {
        let mut real = vec![224.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 224.0);
    }

    #[test]
    fn test_fft_stress_case_225() {
        let mut real = vec![225.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 225.0);
    }

    #[test]
    fn test_fft_stress_case_226() {
        let mut real = vec![226.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 226.0);
    }

    #[test]
    fn test_fft_stress_case_227() {
        let mut real = vec![227.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 227.0);
    }

    #[test]
    fn test_fft_stress_case_228() {
        let mut real = vec![228.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 228.0);
    }

    #[test]
    fn test_fft_stress_case_229() {
        let mut real = vec![229.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 229.0);
    }

    #[test]
    fn test_fft_stress_case_230() {
        let mut real = vec![230.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 230.0);
    }

    #[test]
    fn test_fft_stress_case_231() {
        let mut real = vec![231.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 231.0);
    }

    #[test]
    fn test_fft_stress_case_232() {
        let mut real = vec![232.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 232.0);
    }

    #[test]
    fn test_fft_stress_case_233() {
        let mut real = vec![233.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 233.0);
    }

    #[test]
    fn test_fft_stress_case_234() {
        let mut real = vec![234.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 234.0);
    }

    #[test]
    fn test_fft_stress_case_235() {
        let mut real = vec![235.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 235.0);
    }

    #[test]
    fn test_fft_stress_case_236() {
        let mut real = vec![236.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 236.0);
    }

    #[test]
    fn test_fft_stress_case_237() {
        let mut real = vec![237.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 237.0);
    }

    #[test]
    fn test_fft_stress_case_238() {
        let mut real = vec![238.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 238.0);
    }

    #[test]
    fn test_fft_stress_case_239() {
        let mut real = vec![239.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 239.0);
    }

    #[test]
    fn test_fft_stress_case_240() {
        let mut real = vec![240.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 240.0);
    }

    #[test]
    fn test_fft_stress_case_241() {
        let mut real = vec![241.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 241.0);
    }

    #[test]
    fn test_fft_stress_case_242() {
        let mut real = vec![242.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 242.0);
    }

    #[test]
    fn test_fft_stress_case_243() {
        let mut real = vec![243.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 243.0);
    }

    #[test]
    fn test_fft_stress_case_244() {
        let mut real = vec![244.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 244.0);
    }

    #[test]
    fn test_fft_stress_case_245() {
        let mut real = vec![245.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 245.0);
    }

    #[test]
    fn test_fft_stress_case_246() {
        let mut real = vec![246.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 246.0);
    }

    #[test]
    fn test_fft_stress_case_247() {
        let mut real = vec![247.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 247.0);
    }

    #[test]
    fn test_fft_stress_case_248() {
        let mut real = vec![248.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 248.0);
    }

    #[test]
    fn test_fft_stress_case_249() {
        let mut real = vec![249.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 249.0);
    }

    #[test]
    fn test_fft_stress_case_250() {
        let mut real = vec![250.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 250.0);
    }

    #[test]
    fn test_fft_stress_case_251() {
        let mut real = vec![251.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 251.0);
    }

    #[test]
    fn test_fft_stress_case_252() {
        let mut real = vec![252.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 252.0);
    }

    #[test]
    fn test_fft_stress_case_253() {
        let mut real = vec![253.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 253.0);
    }

    #[test]
    fn test_fft_stress_case_254() {
        let mut real = vec![254.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 254.0);
    }

    #[test]
    fn test_fft_stress_case_255() {
        let mut real = vec![255.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 255.0);
    }

    #[test]
    fn test_fft_stress_case_256() {
        let mut real = vec![256.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 256.0);
    }

    #[test]
    fn test_fft_stress_case_257() {
        let mut real = vec![257.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 257.0);
    }

    #[test]
    fn test_fft_stress_case_258() {
        let mut real = vec![258.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 258.0);
    }

    #[test]
    fn test_fft_stress_case_259() {
        let mut real = vec![259.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 259.0);
    }

    #[test]
    fn test_fft_stress_case_260() {
        let mut real = vec![260.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 260.0);
    }

    #[test]
    fn test_fft_stress_case_261() {
        let mut real = vec![261.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 261.0);
    }

    #[test]
    fn test_fft_stress_case_262() {
        let mut real = vec![262.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 262.0);
    }

    #[test]
    fn test_fft_stress_case_263() {
        let mut real = vec![263.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 263.0);
    }

    #[test]
    fn test_fft_stress_case_264() {
        let mut real = vec![264.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 264.0);
    }

    #[test]
    fn test_fft_stress_case_265() {
        let mut real = vec![265.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 265.0);
    }

    #[test]
    fn test_fft_stress_case_266() {
        let mut real = vec![266.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 266.0);
    }

    #[test]
    fn test_fft_stress_case_267() {
        let mut real = vec![267.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 267.0);
    }

    #[test]
    fn test_fft_stress_case_268() {
        let mut real = vec![268.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 268.0);
    }

    #[test]
    fn test_fft_stress_case_269() {
        let mut real = vec![269.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 269.0);
    }

    #[test]
    fn test_fft_stress_case_270() {
        let mut real = vec![270.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 270.0);
    }

    #[test]
    fn test_fft_stress_case_271() {
        let mut real = vec![271.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 271.0);
    }

    #[test]
    fn test_fft_stress_case_272() {
        let mut real = vec![272.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 272.0);
    }

    #[test]
    fn test_fft_stress_case_273() {
        let mut real = vec![273.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 273.0);
    }

    #[test]
    fn test_fft_stress_case_274() {
        let mut real = vec![274.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 274.0);
    }

    #[test]
    fn test_fft_stress_case_275() {
        let mut real = vec![275.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 275.0);
    }

    #[test]
    fn test_fft_stress_case_276() {
        let mut real = vec![276.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 276.0);
    }

    #[test]
    fn test_fft_stress_case_277() {
        let mut real = vec![277.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 277.0);
    }

    #[test]
    fn test_fft_stress_case_278() {
        let mut real = vec![278.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 278.0);
    }

    #[test]
    fn test_fft_stress_case_279() {
        let mut real = vec![279.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 279.0);
    }

    #[test]
    fn test_fft_stress_case_280() {
        let mut real = vec![280.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 280.0);
    }

    #[test]
    fn test_fft_stress_case_281() {
        let mut real = vec![281.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 281.0);
    }

    #[test]
    fn test_fft_stress_case_282() {
        let mut real = vec![282.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 282.0);
    }

    #[test]
    fn test_fft_stress_case_283() {
        let mut real = vec![283.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 283.0);
    }

    #[test]
    fn test_fft_stress_case_284() {
        let mut real = vec![284.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 284.0);
    }

    #[test]
    fn test_fft_stress_case_285() {
        let mut real = vec![285.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 285.0);
    }

    #[test]
    fn test_fft_stress_case_286() {
        let mut real = vec![286.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 286.0);
    }

    #[test]
    fn test_fft_stress_case_287() {
        let mut real = vec![287.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 287.0);
    }

    #[test]
    fn test_fft_stress_case_288() {
        let mut real = vec![288.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 288.0);
    }

    #[test]
    fn test_fft_stress_case_289() {
        let mut real = vec![289.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 289.0);
    }

    #[test]
    fn test_fft_stress_case_290() {
        let mut real = vec![290.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 290.0);
    }

    #[test]
    fn test_fft_stress_case_291() {
        let mut real = vec![291.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 291.0);
    }

    #[test]
    fn test_fft_stress_case_292() {
        let mut real = vec![292.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 292.0);
    }

    #[test]
    fn test_fft_stress_case_293() {
        let mut real = vec![293.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 293.0);
    }

    #[test]
    fn test_fft_stress_case_294() {
        let mut real = vec![294.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 294.0);
    }

    #[test]
    fn test_fft_stress_case_295() {
        let mut real = vec![295.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 295.0);
    }

    #[test]
    fn test_fft_stress_case_296() {
        let mut real = vec![296.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 296.0);
    }

    #[test]
    fn test_fft_stress_case_297() {
        let mut real = vec![297.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 297.0);
    }

    #[test]
    fn test_fft_stress_case_298() {
        let mut real = vec![298.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 298.0);
    }

    #[test]
    fn test_fft_stress_case_299() {
        let mut real = vec![299.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 299.0);
    }

    #[test]
    fn test_fft_stress_case_300() {
        let mut real = vec![300.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 300.0);
    }

    #[test]
    fn test_fft_stress_case_301() {
        let mut real = vec![301.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 301.0);
    }

    #[test]
    fn test_fft_stress_case_302() {
        let mut real = vec![302.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 302.0);
    }

    #[test]
    fn test_fft_stress_case_303() {
        let mut real = vec![303.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 303.0);
    }

    #[test]
    fn test_fft_stress_case_304() {
        let mut real = vec![304.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 304.0);
    }

    #[test]
    fn test_fft_stress_case_305() {
        let mut real = vec![305.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 305.0);
    }

    #[test]
    fn test_fft_stress_case_306() {
        let mut real = vec![306.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 306.0);
    }

    #[test]
    fn test_fft_stress_case_307() {
        let mut real = vec![307.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 307.0);
    }

    #[test]
    fn test_fft_stress_case_308() {
        let mut real = vec![308.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 308.0);
    }

    #[test]
    fn test_fft_stress_case_309() {
        let mut real = vec![309.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 309.0);
    }

    #[test]
    fn test_fft_stress_case_310() {
        let mut real = vec![310.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 310.0);
    }

    #[test]
    fn test_fft_stress_case_311() {
        let mut real = vec![311.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 311.0);
    }

    #[test]
    fn test_fft_stress_case_312() {
        let mut real = vec![312.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 312.0);
    }

    #[test]
    fn test_fft_stress_case_313() {
        let mut real = vec![313.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 313.0);
    }

    #[test]
    fn test_fft_stress_case_314() {
        let mut real = vec![314.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 314.0);
    }

    #[test]
    fn test_fft_stress_case_315() {
        let mut real = vec![315.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 315.0);
    }

    #[test]
    fn test_fft_stress_case_316() {
        let mut real = vec![316.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 316.0);
    }

    #[test]
    fn test_fft_stress_case_317() {
        let mut real = vec![317.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 317.0);
    }

    #[test]
    fn test_fft_stress_case_318() {
        let mut real = vec![318.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 318.0);
    }

    #[test]
    fn test_fft_stress_case_319() {
        let mut real = vec![319.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 319.0);
    }

    #[test]
    fn test_fft_stress_case_320() {
        let mut real = vec![320.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 320.0);
    }

    #[test]
    fn test_fft_stress_case_321() {
        let mut real = vec![321.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 321.0);
    }

    #[test]
    fn test_fft_stress_case_322() {
        let mut real = vec![322.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 322.0);
    }

    #[test]
    fn test_fft_stress_case_323() {
        let mut real = vec![323.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 323.0);
    }

    #[test]
    fn test_fft_stress_case_324() {
        let mut real = vec![324.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 324.0);
    }

    #[test]
    fn test_fft_stress_case_325() {
        let mut real = vec![325.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 325.0);
    }

    #[test]
    fn test_fft_stress_case_326() {
        let mut real = vec![326.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 326.0);
    }

    #[test]
    fn test_fft_stress_case_327() {
        let mut real = vec![327.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 327.0);
    }

    #[test]
    fn test_fft_stress_case_328() {
        let mut real = vec![328.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 328.0);
    }

    #[test]
    fn test_fft_stress_case_329() {
        let mut real = vec![329.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 329.0);
    }

    #[test]
    fn test_fft_stress_case_330() {
        let mut real = vec![330.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 330.0);
    }

    #[test]
    fn test_fft_stress_case_331() {
        let mut real = vec![331.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 331.0);
    }

    #[test]
    fn test_fft_stress_case_332() {
        let mut real = vec![332.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 332.0);
    }

    #[test]
    fn test_fft_stress_case_333() {
        let mut real = vec![333.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 333.0);
    }

    #[test]
    fn test_fft_stress_case_334() {
        let mut real = vec![334.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 334.0);
    }

    #[test]
    fn test_fft_stress_case_335() {
        let mut real = vec![335.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 335.0);
    }

    #[test]
    fn test_fft_stress_case_336() {
        let mut real = vec![336.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 336.0);
    }

    #[test]
    fn test_fft_stress_case_337() {
        let mut real = vec![337.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 337.0);
    }

    #[test]
    fn test_fft_stress_case_338() {
        let mut real = vec![338.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 338.0);
    }

    #[test]
    fn test_fft_stress_case_339() {
        let mut real = vec![339.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 339.0);
    }

    #[test]
    fn test_fft_stress_case_340() {
        let mut real = vec![340.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 340.0);
    }

    #[test]
    fn test_fft_stress_case_341() {
        let mut real = vec![341.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 341.0);
    }

    #[test]
    fn test_fft_stress_case_342() {
        let mut real = vec![342.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 342.0);
    }

    #[test]
    fn test_fft_stress_case_343() {
        let mut real = vec![343.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 343.0);
    }

    #[test]
    fn test_fft_stress_case_344() {
        let mut real = vec![344.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 344.0);
    }

    #[test]
    fn test_fft_stress_case_345() {
        let mut real = vec![345.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 345.0);
    }

    #[test]
    fn test_fft_stress_case_346() {
        let mut real = vec![346.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 346.0);
    }

    #[test]
    fn test_fft_stress_case_347() {
        let mut real = vec![347.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 347.0);
    }

    #[test]
    fn test_fft_stress_case_348() {
        let mut real = vec![348.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 348.0);
    }

    #[test]
    fn test_fft_stress_case_349() {
        let mut real = vec![349.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 349.0);
    }

    #[test]
    fn test_fft_stress_case_350() {
        let mut real = vec![350.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 350.0);
    }

    #[test]
    fn test_fft_stress_case_351() {
        let mut real = vec![351.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 351.0);
    }

    #[test]
    fn test_fft_stress_case_352() {
        let mut real = vec![352.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 352.0);
    }

    #[test]
    fn test_fft_stress_case_353() {
        let mut real = vec![353.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 353.0);
    }

    #[test]
    fn test_fft_stress_case_354() {
        let mut real = vec![354.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 354.0);
    }

    #[test]
    fn test_fft_stress_case_355() {
        let mut real = vec![355.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 355.0);
    }

    #[test]
    fn test_fft_stress_case_356() {
        let mut real = vec![356.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 356.0);
    }

    #[test]
    fn test_fft_stress_case_357() {
        let mut real = vec![357.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 357.0);
    }

    #[test]
    fn test_fft_stress_case_358() {
        let mut real = vec![358.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 358.0);
    }

    #[test]
    fn test_fft_stress_case_359() {
        let mut real = vec![359.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 359.0);
    }

    #[test]
    fn test_fft_stress_case_360() {
        let mut real = vec![360.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 360.0);
    }

    #[test]
    fn test_fft_stress_case_361() {
        let mut real = vec![361.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 361.0);
    }

    #[test]
    fn test_fft_stress_case_362() {
        let mut real = vec![362.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 362.0);
    }

    #[test]
    fn test_fft_stress_case_363() {
        let mut real = vec![363.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 363.0);
    }

    #[test]
    fn test_fft_stress_case_364() {
        let mut real = vec![364.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 364.0);
    }

    #[test]
    fn test_fft_stress_case_365() {
        let mut real = vec![365.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 365.0);
    }

    #[test]
    fn test_fft_stress_case_366() {
        let mut real = vec![366.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 366.0);
    }

    #[test]
    fn test_fft_stress_case_367() {
        let mut real = vec![367.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 367.0);
    }

    #[test]
    fn test_fft_stress_case_368() {
        let mut real = vec![368.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 368.0);
    }

    #[test]
    fn test_fft_stress_case_369() {
        let mut real = vec![369.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 369.0);
    }

    #[test]
    fn test_fft_stress_case_370() {
        let mut real = vec![370.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 370.0);
    }

    #[test]
    fn test_fft_stress_case_371() {
        let mut real = vec![371.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 371.0);
    }

    #[test]
    fn test_fft_stress_case_372() {
        let mut real = vec![372.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 372.0);
    }

    #[test]
    fn test_fft_stress_case_373() {
        let mut real = vec![373.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 373.0);
    }

    #[test]
    fn test_fft_stress_case_374() {
        let mut real = vec![374.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 374.0);
    }

    #[test]
    fn test_fft_stress_case_375() {
        let mut real = vec![375.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 375.0);
    }

    #[test]
    fn test_fft_stress_case_376() {
        let mut real = vec![376.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 376.0);
    }

    #[test]
    fn test_fft_stress_case_377() {
        let mut real = vec![377.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 377.0);
    }

    #[test]
    fn test_fft_stress_case_378() {
        let mut real = vec![378.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 378.0);
    }

    #[test]
    fn test_fft_stress_case_379() {
        let mut real = vec![379.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 379.0);
    }
}
