//! # Dueling Deep Q-Networks (Dueling DQN)
//!
//! Decomposes action-values into separate State Value V(s) and Advantage streams A(s, a).
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;

/// Dueling Q-Network stream estimator.
#[derive(Debug, Clone)]
pub struct DuelingQNet {
    pub input_dim: usize,
    pub num_actions: usize,
    pub v_weights: Vec<f64>,
    pub v_bias: f64,
    pub a_weights: Vec<f64>,
    pub a_biases: Vec<f64>,
}

impl DuelingQNet {
    pub fn new(input_dim: usize, num_actions: usize) -> Self {
        Self {
            input_dim,
            num_actions,
            v_weights: vec![0.0; input_dim],
            v_bias: 0.0,
            a_weights: vec![0.0; input_dim * num_actions],
            a_biases: vec![0.0; num_actions],
        }
    }

    pub fn forward(&self, state: &Tensor) -> Vec<f64> {
        let d = state.data();
        let mut v = self.v_bias;
        for i in 0..d.len().min(self.input_dim) {
            v += d[i] * self.v_weights[i];
        }

        let mut a = self.a_biases.clone();
        let mut mean_a = 0.0;
        for act in 0..self.num_actions {
            for i in 0..d.len().min(self.input_dim) {
                a[act] += d[i] * self.a_weights[act * self.input_dim + i];
            }
            mean_a += a[act];
        }
        mean_a /= self.num_actions as f64;

        let mut q = vec![0.0; self.num_actions];
        for act in 0..self.num_actions {
            q[act] = v + (a[act] - mean_a);
        }
        q
    }
}

/// Dueling DQN Agent.
#[derive(Debug, Clone)]
pub struct DuelingDqnAgent {
    pub q_net: DuelingQNet,
}

impl DuelingDqnAgent {
    pub fn new(input_dim: usize, num_actions: usize) -> Self {
        Self {
            q_net: DuelingQNet::new(input_dim, num_actions),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision)]
    use super::*;
    use crate::core::*;
    use crate::env::*;
    use crate::policy::*;
    use crate::value::*;
    use crate::buffer::*;
    use crate::dqn::*;
    use crate::ppo::*;
    use crate::a2c::*;
    use crate::actor_critic::*;
    use crate::sac::*;
    use crate::agents::*;
    use crate::trainer::*;
    use crate::eval::*;
    use crate::checkpoint::*;
    use crate::utils::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
