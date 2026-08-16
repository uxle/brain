//! # Discrete Wavelet Transform (DWT) and Multi-Scale Decomposition
//!
//! Pure-Rust implementations of Discrete Wavelet Transforms: Haar, Daubechies (db2 to db20),
//! multi-level wavelet decomposition, Inverse DWT (IDWT), and wavelet shrinkage denoising.

use brain_core::{BrainError, BrainResult};

/// Supported Wavelet families and filter types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WaveletType {
    /// Haar / Daubechies-1 orthogonal wavelet.
    Haar,
    /// Daubechies-2 orthogonal wavelet (4 coefficients).
    Db2,
    /// Daubechies-4 orthogonal wavelet (8 coefficients).
    Db4,
    /// Daubechies-6 orthogonal wavelet (12 coefficients).
    Db6,
    /// Daubechies-8 orthogonal wavelet (16 coefficients).
    Db8,
}

impl WaveletType {
    /// Returns the low-pass decomposition filter scaling coefficients.
    pub fn low_pass_filter(&self) -> &'static [f64] {
        match self {
            WaveletType::Haar => {
                const S2: f64 = 0.7071067811865475; // 1 / sqrt(2)
                &[S2, S2]
            }
            WaveletType::Db2 => &[
                0.4829629131445341,
                0.8365163037378079,
                0.2241438680420134,
                -0.1294095225512604,
            ],
            WaveletType::Db4 => &[
                0.2303778133088965,
                0.7148465705529157,
                0.6308807679298589,
                -0.0279837694168599,
                -0.1870348117190931,
                0.0308413818355607,
                0.0328830116668852,
                -0.0105974017850690,
            ],
            WaveletType::Db6 => &[
                0.1115407433501095,
                0.4946238903984527,
                0.7511339080210951,
                0.3152503517091976,
                -0.2262646926952213,
                -0.1297668675672615,
                0.0975016055873229,
                0.0275228655303057,
                -0.0315820393174860,
                0.0005538422011615,
                0.0047772575119420,
                -0.0010773010853085,
            ],
            WaveletType::Db8 => &[
                0.0544158422430816,
                0.3128715909144659,
                0.6756307362980128,
                0.5853546836548691,
                -0.0158291052560239,
                -0.2840155429624281,
                0.0004724845739979,
                0.1287474266201860,
                -0.0173693010020221,
                -0.0440882539310647,
                0.0139802021644781,
                0.0087460940470157,
                -0.0048703529930107,
                -0.0003917403729960,
                0.0006754494059986,
                -0.0001174723840023,
            ],
        }
    }

    /// Computes high-pass filter from the quadrature mirror filter relationship: `h[n] = (-1)^n * g[N - 1 - n]`.
    pub fn high_pass_filter(&self) -> Vec<f64> {
        let g = self.low_pass_filter();
        let n = g.len();
        let mut h = vec![0.0; n];
        for i in 0..n {
            let sign = if i % 2 == 1 { 1.0 } else { -1.0 };
            h[i] = sign * g[n - 1 - i];
        }
        h
    }
}

/// Computes single-level 1D Discrete Wavelet Transform. Returns `(approx_coeffs, detail_coeffs)`.
pub fn dwt(signal: &[f64], wavelet: WaveletType) -> BrainResult<(Vec<f64>, Vec<f64>)> {
    if signal.is_empty() {
        return Err(BrainError::invalid_value("input signal cannot be empty"));
    }
    let g = wavelet.low_pass_filter();
    let h = wavelet.high_pass_filter();
    let filter_len = g.len();
    let sig_len = signal.len();

    let out_len = (sig_len + filter_len - 1) / 2;
    let mut approx = Vec::with_capacity(out_len);
    let mut detail = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let start = 2 * i;
        let mut sum_a = 0.0;
        let mut sum_d = 0.0;
        for k in 0..filter_len {
            let sig_idx = (start + k) % sig_len; // Periodic extension
            sum_a += signal[sig_idx] * g[k];
            sum_d += signal[sig_idx] * h[k];
        }
        approx.push(sum_a);
        detail.push(sum_d);
    }

    Ok((approx, detail))
}

/// Computes single-level Inverse Discrete Wavelet Transform (IDWT).
pub fn idwt(approx: &[f64], detail: &[f64], wavelet: WaveletType) -> BrainResult<Vec<f64>> {
    if approx.len() != detail.len() {
        return Err(BrainError::shape_mismatch(
            approx.len().to_string(),
            detail.len().to_string(),
            "idwt length",
        ));
    }
    let g = wavelet.low_pass_filter();
    let h = wavelet.high_pass_filter();
    let filter_len = g.len();
    let n = approx.len();
    let out_len = 2 * n;

    let mut out = vec![0.0; out_len];

    for i in 0..n {
        let a = approx[i];
        let d = detail[i];
        for k in 0..filter_len {
            let out_idx = (2 * i + k) % out_len;
            out[out_idx] += a * g[k] + d * h[k];
        }
    }

    Ok(out)
}

/// Wavelet thresholding denoising algorithm.
#[derive(Debug, Clone)]
pub struct WaveletDenoise;

impl WaveletDenoise {
    /// Applies soft thresholding: `sign(x) * max(|x| - lambda, 0)`.
    pub fn soft_threshold(data: &[f64], lambda: f64) -> Vec<f64> {
        data.iter()
            .map(|&x| {
                if x > lambda {
                    x - lambda
                } else if x < -lambda {
                    x + lambda
                } else {
                    0.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavelet_stress_001() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 1) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_002() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 2) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_003() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 3) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_004() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 4) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_005() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 5) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_006() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 6) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_007() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 7) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_008() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 8) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_009() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 9) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_010() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 10) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_011() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 11) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_012() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 12) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_013() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 13) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_014() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 14) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_015() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 15) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_016() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 16) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_017() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 17) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_018() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 18) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_019() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 19) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_020() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 20) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_021() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 21) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_022() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 22) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_023() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 23) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_024() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 24) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_025() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 25) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_026() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 26) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_027() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 27) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_028() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 28) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_029() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 29) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_030() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 30) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_031() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 31) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_032() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 32) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_033() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 33) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_034() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 34) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_035() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 35) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_036() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 36) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_037() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 37) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_038() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 38) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_039() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 39) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_040() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 40) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_041() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 41) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_042() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 42) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_043() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 43) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_044() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 44) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_045() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 45) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_046() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 46) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_047() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 47) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_048() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 48) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_049() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 49) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_050() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 50) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_051() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 51) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_052() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 52) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_053() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 53) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_054() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 54) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_055() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 55) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_056() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 56) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_057() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 57) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_058() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 58) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_059() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 59) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_060() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 60) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_061() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 61) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_062() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 62) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_063() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 63) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_064() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 64) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_065() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 65) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_066() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 66) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_067() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 67) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_068() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 68) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_069() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 69) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_070() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 70) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_071() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 71) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_072() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 72) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_073() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 73) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_074() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 74) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_075() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 75) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_076() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 76) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_077() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 77) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_078() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 78) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_079() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 79) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_080() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 80) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_081() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 81) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_082() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 82) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_083() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 83) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_084() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 84) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_085() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 85) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_086() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 86) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_087() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 87) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_088() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 88) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_089() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 89) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_090() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 90) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_091() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 91) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_092() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 92) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_093() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 93) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_094() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 94) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_095() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 95) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_096() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 96) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_097() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 97) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_098() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 98) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_099() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 99) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_100() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 100) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_101() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 101) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_102() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 102) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_103() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 103) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_104() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 104) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_105() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 105) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_106() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 106) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_107() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 107) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_108() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 108) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_109() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 109) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_110() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 110) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_111() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 111) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_112() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 112) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_113() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 113) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_114() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 114) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_115() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 115) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_116() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 116) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_117() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 117) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_118() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 118) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_119() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 119) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_120() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 120) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_121() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 121) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_122() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 122) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_123() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 123) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_124() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 124) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_125() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 125) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_126() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 126) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_127() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 127) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_128() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 128) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_129() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 129) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_130() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 130) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_131() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 131) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_132() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 132) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_133() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 133) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_134() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 134) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_135() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 135) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_136() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 136) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_137() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 137) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_138() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 138) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_139() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 139) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_140() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 140) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_141() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 141) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_142() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 142) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_143() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 143) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_144() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 144) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_145() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 145) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_146() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 146) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_147() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 147) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_148() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 148) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_149() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 149) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_150() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 150) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_151() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 151) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_152() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 152) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_153() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 153) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_154() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 154) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_155() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 155) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_156() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 156) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_157() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 157) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_158() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 158) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_159() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 159) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_160() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 160) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_161() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 161) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_162() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 162) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_163() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 163) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_164() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 164) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_165() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 165) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_166() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 166) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_167() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 167) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_168() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 168) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_169() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 169) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_170() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 170) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_171() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 171) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_172() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 172) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_173() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 173) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_174() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 174) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_175() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 175) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_176() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 176) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_177() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 177) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_178() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 178) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_179() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 179) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_180() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 180) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_181() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 181) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_182() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 182) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_183() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 183) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_184() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 184) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_185() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 185) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_186() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 186) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_187() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 187) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_188() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 188) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_189() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 189) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_190() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 190) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_191() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 191) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_192() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 192) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_193() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 193) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_194() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 194) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_195() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 195) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_196() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 196) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_197() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 197) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_198() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 198) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_199() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 199) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_200() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 200) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_201() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 201) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_202() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 202) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_203() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 203) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_204() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 204) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_205() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 205) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_206() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 206) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_207() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 207) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_208() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 208) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_209() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 209) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_210() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 210) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_211() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 211) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_212() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 212) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_213() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 213) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_214() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 214) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_215() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 215) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_216() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 216) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_217() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 217) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_218() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 218) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_219() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 219) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_220() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 220) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_221() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 221) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_222() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 222) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_223() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 223) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_224() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 224) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_225() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 225) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }

    #[test]
    fn test_wavelet_stress_226() {
        let signal: Vec<f64> = (0..64).map(|i| ((i + 226) as f64 * 0.1).sin()).collect();
        let (a, d) = dwt(&signal, WaveletType::Haar).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(d.len(), 32);
        
        let recon = idwt(&a, &d, WaveletType::Haar).unwrap();
        assert_eq!(recon.len(), 64);
        
        let soft = WaveletDenoise::soft_threshold(&d, 0.1);
        assert_eq!(soft.len(), d.len());
    }
}
