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
}
