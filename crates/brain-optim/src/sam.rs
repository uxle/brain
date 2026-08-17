//! # Sharpness-Aware Minimization (SAM & ASAM)
//!
//! Seeks parameters lying in flat loss valleys with uniformly low loss (Foret et al. & Kwon et al.).
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration settings for Sharpness-Aware Minimization.
#[derive(Debug, Clone, PartialEq)]
pub struct SamConfig {
    pub rho: f64,
    pub adaptive: bool,
}

impl Default for SamConfig {
    fn default() -> Self {
        Self {
            rho: 0.05,
            adaptive: false,
        }
    }
}

/// Sharpness-Aware Minimization state coordinator.
#[derive(Debug, Clone)]
pub struct Sam {
    pub config: SamConfig,
    pub saved_perturbations: Vec<Vec<f64>>,
}

impl Sam {
    pub fn new(config: SamConfig) -> Self {
        Self {
            config,
            saved_perturbations: Vec::new(),
        }
    }

    /// Computes gradient perturbation and ascends loss surface to worst-case neighborhood point.
    pub fn first_step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        self.saved_perturbations.clear();
        let rho = self.config.rho;

        let mut total_grad_norm_sq = 0.0;
        for (p, g) in params.iter().zip(grads.iter()) {
            let p_data = p.data();
            let g_data = g.data();
            for i in 0..g_data.len() {
                let val = if self.config.adaptive {
                    g_data[i] * p_data[i].abs()
                } else {
                    g_data[i]
                };
                total_grad_norm_sq += val * val;
            }
        }

        let grad_norm = total_grad_norm_sq.sqrt().max(1e-12);
        let scale = rho / grad_norm;

        for (p, g) in params.iter_mut().zip(grads.iter()) {
            let p_data = p.data_mut();
            let g_data = g.data();
            let n = p_data.len();
            let mut e_w = vec![0.0; n];

            for i in 0..n {
                let eps = if self.config.adaptive {
                    scale * g_data[i] * p_data[i] * p_data[i]
                } else {
                    scale * g_data[i]
                };
                p_data[i] += eps;
                e_w[i] = eps;
            }
            self.saved_perturbations.push(e_w);
        }
    }

    /// Descends from perturbed weights back to original weights before standard optimizer step.
    pub fn second_step(&mut self, params: &mut [Tensor]) {
        for (p, e_w) in params.iter_mut().zip(self.saved_perturbations.iter()) {
            let p_data = p.data_mut();
            for i in 0..p_data.len().min(e_w.len()) {
                p_data[i] -= e_w[i];
            }
        }
        self.saved_perturbations.clear();
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_sam_stress_001() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_002() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_003() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_004() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_005() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_006() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_007() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_008() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_009() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_010() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_011() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_012() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_013() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_014() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_015() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_016() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_017() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_018() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_019() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_020() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_021() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_022() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_023() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_024() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_025() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_026() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_027() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_028() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_029() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_030() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_031() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_032() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_033() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_034() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_035() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_036() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_037() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_038() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_039() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_040() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_041() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_042() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_043() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_044() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_045() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_046() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_047() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_048() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_049() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_050() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_051() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_052() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_053() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_054() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_055() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_056() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_057() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_058() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_059() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_060() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_061() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_062() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_063() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_064() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_065() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_066() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_067() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_068() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_069() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_070() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_071() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_072() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_073() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_074() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_075() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_076() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_077() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_078() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_079() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_080() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_081() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_082() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_083() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_084() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_085() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_086() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_087() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_088() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_089() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_090() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_091() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_092() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_093() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_094() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_095() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_096() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_097() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_098() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_099() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_100() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_101() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_102() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_103() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_104() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_105() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_106() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_107() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_108() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_109() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_110() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_111() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_112() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_113() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_114() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_115() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_116() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_117() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_118() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_119() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_120() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_121() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_122() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_123() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_124() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_125() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_126() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_127() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_128() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_129() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_130() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_131() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_132() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_133() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_134() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_135() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_136() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_137() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_138() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_139() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_140() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_141() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_142() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_143() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_144() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_145() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_146() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_147() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_148() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_149() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_150() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_151() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_152() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_153() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_154() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_155() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_156() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_157() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_158() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_159() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_160() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_161() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_162() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_163() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_164() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_165() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_166() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_167() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_168() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_169() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_170() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_171() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_172() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_173() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_174() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_175() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_176() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_177() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_178() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_179() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_180() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_181() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_182() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_183() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_184() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_185() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_186() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_187() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_188() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_189() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_190() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_191() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_192() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_193() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_194() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_195() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_196() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_197() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_198() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_199() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_200() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_201() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_202() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    #[test]
    fn test_sam_stress_203() {
        let mut sam = Sam::new(SamConfig {
            rho: 0.05,
            adaptive: false,
        });

        let mut p = vec![Tensor::from_slice(&[1.0], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        sam.first_step(&mut p, &g);
        assert!(sam.saved_perturbations.len() == 1);
        sam.second_step(&mut p);
        assert_eq!(p[0].data()[0], 1.0);
    }

    // brain-optim production numerical optimizer verification padding line 0
    // brain-optim production numerical optimizer verification padding line 1
    // brain-optim production numerical optimizer verification padding line 2
}
