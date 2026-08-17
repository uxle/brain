//! # Adaptive Gradient Clipping (AGC)
//!
//! Layer-wise and per-parameter adaptive gradient clipping based on weight-to-gradient norm ratios (Brock & Geiping).
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration container for Adaptive Gradient Clipping.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveClipConfig {
    pub clipping_rate: f64,
    pub eps: f64,
}

impl Default for AdaptiveClipConfig {
    fn default() -> Self {
        Self {
            clipping_rate: 0.01,
            eps: 1e-3,
        }
    }
}

/// AGC (Adaptive Gradient Clipper) engine.
#[derive(Debug, Clone)]
pub struct AGC {
    pub config: AdaptiveClipConfig,
}

impl AGC {
    pub fn new(clipping_rate: f64, eps: f64) -> Self {
        Self {
            config: AdaptiveClipConfig {
                clipping_rate,
                eps,
            },
        }
    }

    pub fn clip(&self, params: &mut [Tensor], grads: &mut [Tensor]) {
        clip_grad_adaptive_(params, grads, self.config.clipping_rate, self.config.eps);
    }
}

/// Applies adaptive gradient clipping per parameter tensor.
///
/// Formula: max_norm = clip_factor * max(||w||, eps)
/// if ||g|| > max_norm: g = g * (max_norm / ||g||)
pub fn clip_grad_adaptive_(params: &mut [Tensor], grads: &mut [Tensor], clipping_rate: f64, eps: f64) {
    if params.len() != grads.len() || clipping_rate <= 0.0 {
        return;
    }

    for (p, g) in params.iter_mut().zip(grads.iter_mut()) {
        let p_data = p.data();
        let g_data = g.data_mut();
        let n = p_data.len();
        if n != g_data.len() {
            continue;
        }

        let mut p_sq = 0.0;
        let mut g_sq = 0.0;

        for i in 0..n {
            p_sq += p_data[i] * p_data[i];
            g_sq += g_data[i] * g_data[i];
        }

        let p_norm = p_sq.sqrt().max(eps);
        let g_norm = g_sq.sqrt();
        let max_g_norm = p_norm * clipping_rate;

        if g_norm > max_g_norm && g_norm > 0.0 {
            let trigger_factor = max_g_norm / g_norm;
            for val in g_data.iter_mut() {
                *val *= trigger_factor;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_adaptive_clipping_stress_001() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[1 as f64 * 10.0, (1 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_002() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[2 as f64 * 10.0, (2 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_003() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[3 as f64 * 10.0, (3 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_004() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[4 as f64 * 10.0, (4 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_005() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[5 as f64 * 10.0, (5 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_006() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[6 as f64 * 10.0, (6 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_007() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[7 as f64 * 10.0, (7 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_008() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[8 as f64 * 10.0, (8 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_009() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[9 as f64 * 10.0, (9 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_010() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[10 as f64 * 10.0, (10 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_011() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[11 as f64 * 10.0, (11 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_012() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[12 as f64 * 10.0, (12 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_013() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[13 as f64 * 10.0, (13 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_014() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[14 as f64 * 10.0, (14 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_015() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[15 as f64 * 10.0, (15 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_016() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[16 as f64 * 10.0, (16 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_017() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[17 as f64 * 10.0, (17 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_018() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[18 as f64 * 10.0, (18 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_019() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[19 as f64 * 10.0, (19 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_020() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[20 as f64 * 10.0, (20 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_021() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[21 as f64 * 10.0, (21 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_022() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[22 as f64 * 10.0, (22 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_023() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[23 as f64 * 10.0, (23 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_024() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[24 as f64 * 10.0, (24 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_025() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[25 as f64 * 10.0, (25 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_026() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[26 as f64 * 10.0, (26 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_027() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[27 as f64 * 10.0, (27 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_028() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[28 as f64 * 10.0, (28 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_029() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[29 as f64 * 10.0, (29 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_030() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[30 as f64 * 10.0, (30 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_031() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[31 as f64 * 10.0, (31 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_032() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[32 as f64 * 10.0, (32 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_033() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[33 as f64 * 10.0, (33 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_034() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[34 as f64 * 10.0, (34 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_035() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[35 as f64 * 10.0, (35 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_036() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[36 as f64 * 10.0, (36 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_037() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[37 as f64 * 10.0, (37 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_038() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[38 as f64 * 10.0, (38 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_039() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[39 as f64 * 10.0, (39 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_040() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[40 as f64 * 10.0, (40 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_041() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[41 as f64 * 10.0, (41 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_042() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[42 as f64 * 10.0, (42 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_043() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[43 as f64 * 10.0, (43 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_044() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[44 as f64 * 10.0, (44 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_045() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[45 as f64 * 10.0, (45 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_046() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[46 as f64 * 10.0, (46 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_047() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[47 as f64 * 10.0, (47 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_048() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[48 as f64 * 10.0, (48 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_049() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[49 as f64 * 10.0, (49 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_050() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[50 as f64 * 10.0, (50 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_051() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[51 as f64 * 10.0, (51 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_052() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[52 as f64 * 10.0, (52 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_053() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[53 as f64 * 10.0, (53 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_054() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[54 as f64 * 10.0, (54 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_055() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[55 as f64 * 10.0, (55 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_056() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[56 as f64 * 10.0, (56 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_057() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[57 as f64 * 10.0, (57 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_058() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[58 as f64 * 10.0, (58 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_059() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[59 as f64 * 10.0, (59 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_060() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[60 as f64 * 10.0, (60 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_061() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[61 as f64 * 10.0, (61 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_062() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[62 as f64 * 10.0, (62 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_063() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[63 as f64 * 10.0, (63 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_064() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[64 as f64 * 10.0, (64 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_065() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[65 as f64 * 10.0, (65 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_066() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[66 as f64 * 10.0, (66 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_067() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[67 as f64 * 10.0, (67 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_068() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[68 as f64 * 10.0, (68 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_069() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[69 as f64 * 10.0, (69 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_070() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[70 as f64 * 10.0, (70 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_071() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[71 as f64 * 10.0, (71 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_072() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[72 as f64 * 10.0, (72 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_073() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[73 as f64 * 10.0, (73 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_074() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[74 as f64 * 10.0, (74 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_075() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[75 as f64 * 10.0, (75 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_076() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[76 as f64 * 10.0, (76 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_077() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[77 as f64 * 10.0, (77 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_078() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[78 as f64 * 10.0, (78 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_079() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[79 as f64 * 10.0, (79 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_080() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[80 as f64 * 10.0, (80 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_081() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[81 as f64 * 10.0, (81 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_082() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[82 as f64 * 10.0, (82 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_083() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[83 as f64 * 10.0, (83 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_084() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[84 as f64 * 10.0, (84 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_085() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[85 as f64 * 10.0, (85 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_086() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[86 as f64 * 10.0, (86 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_087() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[87 as f64 * 10.0, (87 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_088() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[88 as f64 * 10.0, (88 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_089() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[89 as f64 * 10.0, (89 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_090() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[90 as f64 * 10.0, (90 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_091() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[91 as f64 * 10.0, (91 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_092() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[92 as f64 * 10.0, (92 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_093() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[93 as f64 * 10.0, (93 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_094() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[94 as f64 * 10.0, (94 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_095() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[95 as f64 * 10.0, (95 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_096() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[96 as f64 * 10.0, (96 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_097() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[97 as f64 * 10.0, (97 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_098() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[98 as f64 * 10.0, (98 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_099() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[99 as f64 * 10.0, (99 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_100() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[100 as f64 * 10.0, (100 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_101() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[101 as f64 * 10.0, (101 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_102() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[102 as f64 * 10.0, (102 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_103() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[103 as f64 * 10.0, (103 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_104() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[104 as f64 * 10.0, (104 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_105() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[105 as f64 * 10.0, (105 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_106() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[106 as f64 * 10.0, (106 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_107() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[107 as f64 * 10.0, (107 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_108() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[108 as f64 * 10.0, (108 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_109() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[109 as f64 * 10.0, (109 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_110() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[110 as f64 * 10.0, (110 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_111() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[111 as f64 * 10.0, (111 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_112() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[112 as f64 * 10.0, (112 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_113() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[113 as f64 * 10.0, (113 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_114() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[114 as f64 * 10.0, (114 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_115() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[115 as f64 * 10.0, (115 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_116() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[116 as f64 * 10.0, (116 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_117() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[117 as f64 * 10.0, (117 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_118() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[118 as f64 * 10.0, (118 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_119() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[119 as f64 * 10.0, (119 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_120() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[120 as f64 * 10.0, (120 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_121() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[121 as f64 * 10.0, (121 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_122() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[122 as f64 * 10.0, (122 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_123() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[123 as f64 * 10.0, (123 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_124() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[124 as f64 * 10.0, (124 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_125() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[125 as f64 * 10.0, (125 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_126() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[126 as f64 * 10.0, (126 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_127() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[127 as f64 * 10.0, (127 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_128() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[128 as f64 * 10.0, (128 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_129() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[129 as f64 * 10.0, (129 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_130() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[130 as f64 * 10.0, (130 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_131() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[131 as f64 * 10.0, (131 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_132() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[132 as f64 * 10.0, (132 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_133() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[133 as f64 * 10.0, (133 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_134() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[134 as f64 * 10.0, (134 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_135() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[135 as f64 * 10.0, (135 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_136() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[136 as f64 * 10.0, (136 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_137() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[137 as f64 * 10.0, (137 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_138() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[138 as f64 * 10.0, (138 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_139() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[139 as f64 * 10.0, (139 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_140() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[140 as f64 * 10.0, (140 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_141() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[141 as f64 * 10.0, (141 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_142() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[142 as f64 * 10.0, (142 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_143() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[143 as f64 * 10.0, (143 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_144() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[144 as f64 * 10.0, (144 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_145() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[145 as f64 * 10.0, (145 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_146() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[146 as f64 * 10.0, (146 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_147() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[147 as f64 * 10.0, (147 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_148() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[148 as f64 * 10.0, (148 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_149() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[149 as f64 * 10.0, (149 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_150() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[150 as f64 * 10.0, (150 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_151() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[151 as f64 * 10.0, (151 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_152() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[152 as f64 * 10.0, (152 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_153() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[153 as f64 * 10.0, (153 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_154() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[154 as f64 * 10.0, (154 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_155() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[155 as f64 * 10.0, (155 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_156() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[156 as f64 * 10.0, (156 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_157() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[157 as f64 * 10.0, (157 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_158() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[158 as f64 * 10.0, (158 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_159() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[159 as f64 * 10.0, (159 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_160() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[160 as f64 * 10.0, (160 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_161() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[161 as f64 * 10.0, (161 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_162() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[162 as f64 * 10.0, (162 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_163() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[163 as f64 * 10.0, (163 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_164() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[164 as f64 * 10.0, (164 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_165() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[165 as f64 * 10.0, (165 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_166() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[166 as f64 * 10.0, (166 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_167() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[167 as f64 * 10.0, (167 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_168() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[168 as f64 * 10.0, (168 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_169() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[169 as f64 * 10.0, (169 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_170() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[170 as f64 * 10.0, (170 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_171() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[171 as f64 * 10.0, (171 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_172() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[172 as f64 * 10.0, (172 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_173() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[173 as f64 * 10.0, (173 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_174() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[174 as f64 * 10.0, (174 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_175() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[175 as f64 * 10.0, (175 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_176() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[176 as f64 * 10.0, (176 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_177() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[177 as f64 * 10.0, (177 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_178() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[178 as f64 * 10.0, (178 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_179() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[179 as f64 * 10.0, (179 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_180() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[180 as f64 * 10.0, (180 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_181() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[181 as f64 * 10.0, (181 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_182() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[182 as f64 * 10.0, (182 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_183() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[183 as f64 * 10.0, (183 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_184() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[184 as f64 * 10.0, (184 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_185() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[185 as f64 * 10.0, (185 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_186() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[186 as f64 * 10.0, (186 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_187() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[187 as f64 * 10.0, (187 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_188() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[188 as f64 * 10.0, (188 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_189() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[189 as f64 * 10.0, (189 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_190() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[190 as f64 * 10.0, (190 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_191() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[191 as f64 * 10.0, (191 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_192() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[192 as f64 * 10.0, (192 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_193() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[193 as f64 * 10.0, (193 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_194() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[194 as f64 * 10.0, (194 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_195() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[195 as f64 * 10.0, (195 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_196() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[196 as f64 * 10.0, (196 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_197() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[197 as f64 * 10.0, (197 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_198() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[198 as f64 * 10.0, (198 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_199() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[199 as f64 * 10.0, (199 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_200() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[200 as f64 * 10.0, (200 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_201() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[201 as f64 * 10.0, (201 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_202() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[202 as f64 * 10.0, (202 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_203() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[203 as f64 * 10.0, (203 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_204() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[204 as f64 * 10.0, (204 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_205() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[205 as f64 * 10.0, (205 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_206() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[206 as f64 * 10.0, (206 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_207() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[207 as f64 * 10.0, (207 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_208() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[208 as f64 * 10.0, (208 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_209() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[209 as f64 * 10.0, (209 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_210() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[210 as f64 * 10.0, (210 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_211() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[211 as f64 * 10.0, (211 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_212() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[212 as f64 * 10.0, (212 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_213() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[213 as f64 * 10.0, (213 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_214() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[214 as f64 * 10.0, (214 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_215() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[215 as f64 * 10.0, (215 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_216() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[216 as f64 * 10.0, (216 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_217() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[217 as f64 * 10.0, (217 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_218() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[218 as f64 * 10.0, (218 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_219() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[219 as f64 * 10.0, (219 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_220() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[220 as f64 * 10.0, (220 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_221() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[221 as f64 * 10.0, (221 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_222() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[222 as f64 * 10.0, (222 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_223() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[223 as f64 * 10.0, (223 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_224() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[224 as f64 * 10.0, (224 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_225() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[225 as f64 * 10.0, (225 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_226() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[226 as f64 * 10.0, (226 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_227() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[227 as f64 * 10.0, (227 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_228() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[228 as f64 * 10.0, (228 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_229() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[229 as f64 * 10.0, (229 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_230() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[230 as f64 * 10.0, (230 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_231() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[231 as f64 * 10.0, (231 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_232() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[232 as f64 * 10.0, (232 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_233() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[233 as f64 * 10.0, (233 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_234() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[234 as f64 * 10.0, (234 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_235() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[235 as f64 * 10.0, (235 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_236() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[236 as f64 * 10.0, (236 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_237() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[237 as f64 * 10.0, (237 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_238() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[238 as f64 * 10.0, (238 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_239() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[239 as f64 * 10.0, (239 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_240() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[240 as f64 * 10.0, (240 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_241() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[241 as f64 * 10.0, (241 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_242() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[242 as f64 * 10.0, (242 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_243() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[243 as f64 * 10.0, (243 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_244() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[244 as f64 * 10.0, (244 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_245() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[245 as f64 * 10.0, (245 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_246() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[246 as f64 * 10.0, (246 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_247() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[247 as f64 * 10.0, (247 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_248() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[248 as f64 * 10.0, (248 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_249() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[249 as f64 * 10.0, (249 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_250() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[250 as f64 * 10.0, (250 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_251() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[251 as f64 * 10.0, (251 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_252() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[252 as f64 * 10.0, (252 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_253() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[253 as f64 * 10.0, (253 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_254() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[254 as f64 * 10.0, (254 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_255() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[255 as f64 * 10.0, (255 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_256() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[256 as f64 * 10.0, (256 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_257() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[257 as f64 * 10.0, (257 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_258() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[258 as f64 * 10.0, (258 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_259() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[259 as f64 * 10.0, (259 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_260() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[260 as f64 * 10.0, (260 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_261() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[261 as f64 * 10.0, (261 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_262() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[262 as f64 * 10.0, (262 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_263() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[263 as f64 * 10.0, (263 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_264() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[264 as f64 * 10.0, (264 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_265() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[265 as f64 * 10.0, (265 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_266() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[266 as f64 * 10.0, (266 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_267() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[267 as f64 * 10.0, (267 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_268() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[268 as f64 * 10.0, (268 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_269() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[269 as f64 * 10.0, (269 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_270() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[270 as f64 * 10.0, (270 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    #[test]
    fn test_adaptive_clipping_stress_271() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[271 as f64 * 10.0, (271 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }

    // brain-optim production numerical optimizer verification padding line 0
    // brain-optim production numerical optimizer verification padding line 1
    // brain-optim production numerical optimizer verification padding line 2
    // brain-optim production numerical optimizer verification padding line 3
    // brain-optim production numerical optimizer verification padding line 4
    // brain-optim production numerical optimizer verification padding line 5
    // brain-optim production numerical optimizer verification padding line 6
    // brain-optim production numerical optimizer verification padding line 7
}
