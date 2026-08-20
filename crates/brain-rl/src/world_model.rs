//! # Differentiable World Models
//!
//! Predicts next state $\hat{s}_{t+1}$ and reward $\hat{r}_t$ from current state $s_t$ and action $a_t$.

use brain_core::Tensor;
use crate::core::{RlError, RlResult};

/// Transition and reward prediction from the world model.
#[derive(Debug, Clone)]
pub struct WorldModelPrediction {
    pub next_state: Tensor,
    pub reward: f64,
}

/// Differentiable recurrent / feedforward World Model.
#[derive(Debug, Clone)]
pub struct WorldModel {
    pub state_dim: usize,
    pub action_dim: usize,
    pub hidden_dim: usize,
    /// Weights for state-action embedding: [hidden_dim, state_dim + action_dim]
    pub w_embed: Tensor,
    pub b_embed: Tensor,
    /// Weights for next state transition: [state_dim, hidden_dim]
    pub w_state: Tensor,
    pub b_state: Tensor,
    /// Weights for reward prediction: [1, hidden_dim]
    pub w_reward: Tensor,
    pub b_reward: f64,
}

impl WorldModel {
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        let in_dim = state_dim + action_dim;
        let scale = (2.0 / in_dim as f64).sqrt();
        
        let w_embed_data: Vec<f64> = (0..hidden_dim * in_dim)
            .map(|i| ((i as f64 * 0.1337).sin()) * scale)
            .collect();
        let w_state_data: Vec<f64> = (0..state_dim * hidden_dim)
            .map(|i| ((i as f64 * 0.7331).cos()) * scale)
            .collect();
        let w_reward_data: Vec<f64> = (0..hidden_dim)
            .map(|i| ((i as f64 * 0.4242).sin()) * scale)
            .collect();

        Self {
            state_dim,
            action_dim,
            hidden_dim,
            w_embed: Tensor::from_vec(w_embed_data, vec![hidden_dim, in_dim]),
            b_embed: Tensor::zeros(vec![hidden_dim]),
            w_state: Tensor::from_vec(w_state_data, vec![state_dim, hidden_dim]),
            b_state: Tensor::zeros(vec![state_dim]),
            w_reward: Tensor::from_vec(w_reward_data, vec![1, hidden_dim]),
            b_reward: 0.0,
        }
    }

    /// Forward pass: predicts next state and reward from concatenated [state, action].
    pub fn predict(&self, state: &Tensor, action: &Tensor) -> RlResult<WorldModelPrediction> {
        let s_data = state.data();
        let a_data = action.data();

        if s_data.len() != self.state_dim {
            return Err(RlError::InvalidStateShape { expected: vec![self.state_dim], found: state.shape().to_vec() });
        }
        if a_data.len() != self.action_dim {
            return Err(RlError::InvalidStateShape { expected: vec![self.action_dim], found: action.shape().to_vec() });
        }

        // Concatenate state and action
        let mut sa = Vec::with_capacity(self.state_dim + self.action_dim);
        sa.extend_from_slice(s_data);
        sa.extend_from_slice(a_data);

        // Hidden representation h = ReLU(W_embed * sa + b_embed)
        let in_dim = self.state_dim + self.action_dim;
        let mut h = vec![0.0f64; self.hidden_dim];
        let w_emb = self.w_embed.data();
        let b_emb = self.b_embed.data();

        for i in 0..self.hidden_dim {
            let mut dot = b_emb[i];
            for j in 0..in_dim {
                dot += w_emb[i * in_dim + j] * sa[j];
            }
            h[i] = dot.max(0.0); // ReLU
        }

        // Next state = W_state * h + b_state + residual(state)
        let mut s_next = vec![0.0f64; self.state_dim];
        let w_st = self.w_state.data();
        let b_st = self.b_state.data();

        for i in 0..self.state_dim {
            let mut dot = b_st[i];
            for j in 0..self.hidden_dim {
                dot += w_st[i * self.hidden_dim + j] * h[j];
            }
            s_next[i] = dot + s_data[i]; // Residual connection
        }

        // Reward = W_reward * h + b_reward
        let w_r = self.w_reward.data();
        let mut r = self.b_reward;
        for j in 0..self.hidden_dim {
            r += w_r[j] * h[j];
        }

        Ok(WorldModelPrediction {
            next_state: Tensor::from_vec(s_next, vec![self.state_dim]),
            reward: r,
        })
    }

    /// Computes transition and reward MSE losses for supervised updates.
    pub fn loss(&self, state: &Tensor, action: &Tensor, target_next_state: &Tensor, target_reward: f64) -> RlResult<f64> {
        let pred = self.predict(state, action)?;
        let s_pred = pred.next_state.data();
        let s_target = target_next_state.data();

        let state_mse: f64 = s_pred
            .iter()
            .zip(s_target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>() / (self.state_dim as f64);

        let reward_mse = (pred.reward - target_reward).powi(2);
        Ok(state_mse + 0.5 * reward_mse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_model_prediction() {
        let wm = WorldModel::new(4, 2, 16);
        let s = Tensor::from_slice(&[1.0, 0.5, -0.2, 0.8], vec![4]);
        let a = Tensor::from_slice(&[0.1, -0.5], vec![2]);

        let pred = wm.predict(&s, &a).unwrap();
        assert_eq!(pred.next_state.shape(), &[4]);
        let loss = wm.loss(&s, &a, &pred.next_state, pred.reward).unwrap();
        assert!(loss < 1e-6);
    }
}
