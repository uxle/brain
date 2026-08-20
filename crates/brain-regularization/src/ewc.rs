//! # Elastic Weight Consolidation (EWC)
//!
//! Kirkpatrick et al., 2017: "Overcoming catastrophic forgetting in neural networks".
//! Penalizes updates to parameters critical to previous tasks using the diagonal Fisher Information Matrix.

use brain_core::Tensor;
use std::collections::HashMap;

/// Elastic Weight Consolidation (EWC) regularizer.
#[derive(Debug, Clone)]
pub struct ElasticWeightConsolidation {
    pub lambda: f64,
    /// Reference parameters $\theta^*$ from previous tasks (flattened data per param index).
    pub star_params: HashMap<usize, Vec<f64>>,
    /// Diagonal elements of the empirical Fisher Information matrix $F_i$.
    pub fisher_diag: HashMap<usize, Vec<f64>>,
}

impl ElasticWeightConsolidation {
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            star_params: HashMap::new(),
            fisher_diag: HashMap::new(),
        }
    }

    /// Registers a consolidated task checkpoint with empirical Fisher information.
    pub fn register_task(&mut self, current_params: &[Tensor], empirical_gradients: &[Vec<f64>]) {
        for (idx, p) in current_params.iter().enumerate() {
            let p_data = p.data().to_vec();
            let mut f_diag = vec![1e-4; p_data.len()]; // small prior

            if let Some(grads) = empirical_gradients.get(idx) {
                for (i, &g) in grads.iter().enumerate().take(p_data.len()) {
                    f_diag[i] += g * g; // E[g^2]
                }
            }

            self.star_params.insert(idx, p_data);
            self.fisher_diag.insert(idx, f_diag);
        }
    }

    /// Computes the EWC quadratic penalty: $\mathcal{L}_{EWC} = \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_i^*)^2$.
    pub fn compute_penalty(&self, current_params: &[Tensor]) -> f64 {
        let mut total_penalty = 0.0;

        for (idx, p) in current_params.iter().enumerate() {
            if let (Some(star), Some(fisher)) =
                (self.star_params.get(&idx), self.fisher_diag.get(&idx))
            {
                let p_data = p.data();
                let n = p_data.len().min(star.len()).min(fisher.len());

                for i in 0..n {
                    let diff = p_data[i] - star[i];
                    total_penalty += 0.5 * self.lambda * fisher[i] * diff * diff;
                }
            }
        }

        total_penalty
    }

    /// Computes parameter-wise gradients of the EWC penalty: $\nabla_{\theta_i} \mathcal{L}_{EWC} = \lambda F_i (\theta_i - \theta_i^*)$.
    pub fn compute_gradients(&self, current_params: &[Tensor]) -> Vec<Tensor> {
        let mut grad_tensors = Vec::with_capacity(current_params.len());

        for (idx, p) in current_params.iter().enumerate() {
            let p_data = p.data();
            let mut grad = vec![0.0f64; p_data.len()];

            if let (Some(star), Some(fisher)) =
                (self.star_params.get(&idx), self.fisher_diag.get(&idx))
            {
                let n = p_data.len().min(star.len()).min(fisher.len());
                for i in 0..n {
                    grad[i] = self.lambda * fisher[i] * (p_data[i] - star[i]);
                }
            }

            grad_tensors.push(Tensor::from_vec(grad, p.shape().to_vec()));
        }

        grad_tensors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewc_penalty_and_gradients() {
        let mut ewc = ElasticWeightConsolidation::new(100.0);
        let p1 = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let grads = vec![vec![0.5, 0.5]];

        ewc.register_task(&[p1.clone()], &grads);

        // Perturb parameters
        let p1_new = Tensor::from_slice(&[1.1, 1.9], vec![2]);
        let penalty = ewc.compute_penalty(&[p1_new.clone()]);
        assert!(penalty > 0.0);

        let g = ewc.compute_gradients(&[p1_new]);
        assert_eq!(g.len(), 1);
        assert!(g[0].data()[0] > 0.0); // +0.1 diff -> positive grad
        assert!(g[0].data()[1] < 0.0); // -0.1 diff -> negative grad
    }
}
