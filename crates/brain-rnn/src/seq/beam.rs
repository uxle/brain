//! # Beam Search Sequence Decoding
//!
//! Top-$k$ hypothesis tracking over vocabulary log-probabilities with length penalty.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

/// Single Beam Hypothesis candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct BeamHypothesis {
    pub tokens: Vec<usize>,
    pub log_prob: f64,
    pub is_finished: bool,
}

impl BeamHypothesis {
    pub fn new(initial_token: usize) -> Self {
        Self {
            tokens: vec![initial_token],
            log_prob: 0.0,
            is_finished: false,
        }
    }

    pub fn score(&self, alpha: f64) -> f64 {
        let len_penalty = ((5.0 + self.tokens.len() as f64) / 6.0).powf(alpha);
        self.log_prob / len_penalty.max(1e-6)
    }
}

/// Configuration options for Beam Search.
#[derive(Debug, Clone, PartialEq)]
pub struct BeamConfig {
    pub beam_width: usize,
    pub max_length: usize,
    pub length_penalty_alpha: f64,
    pub eos_token: usize,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_length: 50,
            length_penalty_alpha: 0.6,
            eos_token: 2,
        }
    }
}

/// Beam Search Decoder.
#[derive(Debug, Clone)]
pub struct BeamSearch {
    pub config: BeamConfig,
    pub active_beams: Vec<BeamHypothesis>,
    pub completed_beams: Vec<BeamHypothesis>,
}

impl BeamSearch {
    pub fn new(config: BeamConfig, start_token: usize) -> Self {
        Self {
            config,
            active_beams: vec![BeamHypothesis::new(start_token)],
            completed_beams: Vec::new(),
        }
    }

    /// Performs one step of expansion given top candidate log probabilities for each beam.
    pub fn step(&mut self, next_token_log_probs: &[Vec<f64>]) {
        let mut candidates = Vec::new();

        for (beam_idx, beam) in self.active_beams.iter().enumerate() {
            if beam_idx >= next_token_log_probs.len() {
                continue;
            }
            let probs = &next_token_log_probs[beam_idx];
            for (token_id, &lp) in probs.iter().enumerate() {
                let mut new_tokens = beam.tokens.clone();
                new_tokens.push(token_id);
                let is_eos = token_id == self.config.eos_token;
                candidates.push(BeamHypothesis {
                    tokens: new_tokens,
                    log_prob: beam.log_prob + lp,
                    is_finished: is_eos,
                });
            }
        }

        let alpha = self.config.length_penalty_alpha;
        candidates.sort_by(|a, b| b.score(alpha).partial_cmp(&a.score(alpha)).unwrap());

        let mut next_active = Vec::new();
        for cand in candidates.into_iter().take(self.config.beam_width) {
            if cand.is_finished {
                self.completed_beams.push(cand);
            } else {
                next_active.push(cand);
            }
        }
        self.active_beams = next_active;
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::cells::*;
    use crate::seq::*;
    use crate::init_rnn::*;
    use crate::reg_ops::*;
    use crate::process::*;
    use crate::backward_ops::*;
    use crate::builder::*;
    use crate::helper::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
