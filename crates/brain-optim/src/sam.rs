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
}
