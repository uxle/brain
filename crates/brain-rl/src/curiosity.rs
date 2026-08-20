//! # Intrinsic Curiosity Module (ICM)
//!
//! Pathak et al., 2017: "Curiosity-driven Exploration by Self-supervised Prediction".
//! Generates intrinsic reward from forward-model prediction error in a feature space learned via inverse dynamics.

use brain_core::Tensor;
use crate::core::{RlError, RlResult};

/// Intrinsic Curiosity Module (ICM).
#[derive(Debug, Clone)]
pub struct IntrinsicCuriosityModule {
    pub state_dim: usize,
    pub action_dim: usize,
    pub feature_dim: usize,
    pub eta: f64,
    /// Feature encoder weights: [feature_dim, state_dim]
    pub w_enc: Tensor,
    pub b_enc: Tensor,
    /// Forward dynamics weights: [feature_dim, feature_dim + action_dim]
    pub w_forward: Tensor,
    pub b_forward: Tensor,
}

impl IntrinsicCuriosityModule {
    pub fn new(state_dim: usize, action_dim: usize, feature_dim: usize, eta: f64) -> Self {
        let scale = (2.0 / state_dim as f64).sqrt();
        let in_fwd = feature_dim + action_dim;

        let w_enc_data: Vec<f64> = (0..feature_dim * state_dim)
            .map(|i| ((i as f64 * 0.177).sin()) * scale)
            .collect();
        let w_fwd_data: Vec<f64> = (0..feature_dim * in_fwd)
            .map(|i| ((i as f64 * 0.288).cos()) * scale)
            .collect();

        Self {
            state_dim,
            action_dim,
            feature_dim,
            eta,
            w_enc: Tensor::from_vec(w_enc_data, vec![feature_dim, state_dim]),
            b_enc: Tensor::zeros(vec![feature_dim]),
            w_forward: Tensor::from_vec(w_fwd_data, vec![feature_dim, in_fwd]),
            b_forward: Tensor::zeros(vec![feature_dim]),
        }
    }

    /// Encodes raw state $s$ into feature representation $\phi(s) = \text{LeakyReLU}(W_{enc} s + b_{enc})$.
    pub fn encode_state(&self, state: &Tensor) -> RlResult<Tensor> {
        let s_data = state.data();
        if s_data.len() != self.state_dim {
            return Err(RlError::InvalidStateShape { expected: vec![self.state_dim], found: state.shape().to_vec() });
        }

        let mut phi = vec![0.0f64; self.feature_dim];
        let w = self.w_enc.data();
        let b = self.b_enc.data();

        for i in 0..self.feature_dim {
            let mut dot = b[i];
            for j in 0..self.state_dim {
                dot += w[i * self.state_dim + j] * s_data[j];
            }
            phi[i] = if dot >= 0.0 { dot } else { 0.01 * dot }; // LeakyReLU
        }

        Ok(Tensor::from_vec(phi, vec![self.feature_dim]))
    }

    /// Computes intrinsic curiosity reward: $r_i = \frac{\eta}{2} \|\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\|^2$.
    pub fn compute_intrinsic_reward(&self, state: &Tensor, action: &Tensor, next_state: &Tensor) -> RlResult<f64> {
        let phi_s = self.encode_state(state)?;
        let phi_next = self.encode_state(next_state)?;

        let a_data = action.data();
        if a_data.len() != self.action_dim {
            return Err(RlError::InvalidStateShape { expected: vec![self.action_dim], found: action.shape().to_vec() });
        }

        // Input to forward dynamics: [phi_s, action]
        let mut in_fwd = Vec::with_capacity(self.feature_dim + self.action_dim);
        in_fwd.extend_from_slice(phi_s.data());
        in_fwd.extend_from_slice(a_data);

        let total_in = in_fwd.len();
        let mut phi_pred = vec![0.0f64; self.feature_dim];
        let w_f = self.w_forward.data();
        let b_f = self.b_forward.data();

        for i in 0..self.feature_dim {
            let mut dot = b_f[i];
            for j in 0..total_in {
                dot += w_f[i * total_in + j] * in_fwd[j];
            }
            phi_pred[i] = dot;
        }

        let phi_true = phi_next.data();
        let mut mse = 0.0;
        for i in 0..self.feature_dim {
            mse += (phi_pred[i] - phi_true[i]).powi(2);
        }

        let reward = 0.5 * self.eta * mse;
        Ok(reward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icm_intrinsic_reward() {
        let icm = IntrinsicCuriosityModule::new(4, 2, 8, 0.1);
        let s = Tensor::from_slice(&[1.0, 0.5, -0.2, 0.8], vec![4]);
        let a = Tensor::from_slice(&[0.1, -0.5], vec![2]);
        let s_next = Tensor::from_slice(&[1.1, 0.4, -0.1, 0.7], vec![4]);

        let reward = icm.compute_intrinsic_reward(&s, &a, &s_next).unwrap();
        assert!(reward >= 0.0);
    }
}
