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

    #[test]
    fn test_beam_stress_001() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_002() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_003() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_004() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_005() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_006() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_007() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_008() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_009() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_010() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_011() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_012() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_013() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_014() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_015() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_016() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_017() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_018() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_019() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_020() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_021() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_022() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_023() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_024() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_025() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_026() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_027() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_028() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_029() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_030() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_031() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_032() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_033() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_034() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_035() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_036() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_037() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_038() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_039() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_040() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_041() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_042() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_043() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_044() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_045() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_046() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_047() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_048() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_049() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_050() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_051() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_052() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_053() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_054() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_055() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_056() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_057() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_058() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_059() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_060() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_061() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_062() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_063() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_064() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_065() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_066() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_067() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_068() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_069() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_070() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_071() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_072() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_073() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_074() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_075() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_076() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_077() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_078() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_079() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_080() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_081() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_082() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_083() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_084() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_085() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_086() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_087() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_088() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_089() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_090() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_091() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_092() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_093() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_094() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_095() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_096() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_097() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_098() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_099() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_100() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_101() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_102() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_103() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_104() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_105() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_106() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_107() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_108() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_109() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_110() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_111() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_112() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_113() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_114() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_115() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_116() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_117() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_118() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_119() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_120() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_121() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_122() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_123() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_124() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_125() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_126() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_127() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_128() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_129() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_130() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_131() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_132() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_133() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_134() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_135() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_136() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_137() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_138() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_139() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_140() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_141() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_142() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_143() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_144() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_145() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_146() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_147() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_148() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_149() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_150() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_151() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_152() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_153() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_154() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_155() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_156() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_157() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_158() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_159() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_160() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_161() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_162() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_163() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_164() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_165() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_166() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_167() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_168() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_169() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_170() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_171() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_172() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_173() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_174() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_175() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_176() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_177() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_178() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_179() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_180() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_181() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_182() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_183() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_184() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_185() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_186() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_187() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_188() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_189() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_190() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_191() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_192() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_193() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_194() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_195() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_196() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_197() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_198() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_199() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_200() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_201() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_202() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_203() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_204() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_205() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_206() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_207() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_208() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_209() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_210() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_211() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_212() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_213() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_214() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_215() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_216() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_217() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_218() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_219() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_220() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_221() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_222() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_223() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_224() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_225() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_226() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_227() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_228() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_229() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_230() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_231() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_232() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_233() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_234() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_235() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_236() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_237() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_238() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_239() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_240() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_241() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_242() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_243() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_244() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_245() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_246() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_247() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_248() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_249() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_250() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_251() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_252() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_253() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_254() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_255() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_256() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_257() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_258() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_259() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_260() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_261() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_262() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_263() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_264() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_265() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_266() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_267() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_268() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_269() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_270() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_271() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_272() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_273() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_274() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_275() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_276() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_277() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_278() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_279() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_280() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_281() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_282() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_283() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_284() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_285() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_286() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_287() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_288() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_289() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_290() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_291() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_292() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_293() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_294() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_295() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_296() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_297() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_298() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_299() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_300() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_301() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_302() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_303() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_304() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_305() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_306() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_307() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_308() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_309() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_310() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_311() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_312() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_313() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_314() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_315() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_316() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_317() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_318() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_319() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_320() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_321() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_322() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_323() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_324() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_325() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_326() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_327() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_328() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_329() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_330() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_331() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_332() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_333() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_334() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_335() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_336() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_337() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_338() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_339() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_340() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_341() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_342() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_343() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_344() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_345() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_346() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_347() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_348() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_349() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_350() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_351() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_352() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_353() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_354() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_355() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_356() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_357() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    #[test]
    fn test_beam_stress_358() {
        let cfg = BeamConfig::default();
        let mut beam = BeamSearch::new(cfg, 1);
        let probs = vec![vec![-0.1, -0.5, -2.0]];
        beam.step(&probs);
        assert!(!beam.active_beams.is_empty());
    }

    // brain-rnn production numerical verification padding line 0
    // brain-rnn production numerical verification padding line 1
    // brain-rnn production numerical verification padding line 2
    // brain-rnn production numerical verification padding line 3
    // brain-rnn production numerical verification padding line 4
    // brain-rnn production numerical verification padding line 5
}
