//! # Connectionist Temporal Classification (CTC) Loss
//!
//! Sequence alignment loss without frame-by-frame alignment labels (Graves et al. 2006).
#![allow(missing_docs)]

use crate::core::{LossResult, Reduction};
use crate::utils::reduction_apply;
use brain_core::Tensor;

/// Configuration for CTCLoss.
#[derive(Debug, Clone, PartialEq)]
pub struct CTCConfig {
    pub blank: usize,
    pub reduction: Reduction,
    pub zero_infinity: bool,
}

impl Default for CTCConfig {
    fn default() -> Self {
        Self {
            blank: 0,
            reduction: Reduction::Mean,
            zero_infinity: false,
        }
    }
}

/// Connectionist Temporal Classification (CTC) Loss.
#[derive(Debug, Clone, Default)]
pub struct CTCLoss {
    pub config: CTCConfig,
}

impl CTCLoss {
    pub fn new(config: CTCConfig) -> Self {
        Self { config }
    }

    /// Computes CTC loss given:
    /// - `log_probs`: `(T, N, C)` or `(N, T, C)` log probabilities.
    /// - `targets`: slice of target sequences for each batch sample.
    pub fn forward(
        &self,
        log_probs: &Tensor,
        targets: &[Vec<usize>],
        time_major: bool,
    ) -> LossResult<Tensor> {
        let shape = log_probs.shape();
        assert_eq!(
            shape.len(),
            3,
            "CTCLoss expects 3D log_probs [T, N, C] or [N, T, C]"
        );

        let (t_len, batch_size, num_classes) = if time_major {
            (shape[0], shape[1], shape[2])
        } else {
            (shape[1], shape[0], shape[2])
        };

        let mut sample_losses = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let target_seq = if b < targets.len() {
                &targets[b]
            } else {
                &[][..]
            };

            let u = target_seq.len();
            if u == 0 {
                // If target sequence is empty, only all-blank alignment is valid
                let mut log_p = 0.0;
                for t in 0..t_len {
                    let lp = if time_major {
                        log_probs.get_3d(t, b, self.config.blank)
                    } else {
                        log_probs.get_3d(b, t, self.config.blank)
                    };
                    log_p += lp;
                }
                sample_losses.push(-log_p);
                continue;
            }

            // Build extended target sequence with blanks: S = 2U + 1
            let s_len = 2 * u + 1;
            let mut l_prime = vec![self.config.blank; s_len];
            for i in 0..u {
                l_prime[2 * i + 1] = target_seq[i];
            }

            if t_len < u {
                // Time steps fewer than targets -> impossible alignment
                let loss = if self.config.zero_infinity {
                    0.0
                } else {
                    f64::INFINITY
                };
                sample_losses.push(loss);
                continue;
            }

            // Forward DP table in log space: alpha[t, s]
            let mut alpha = vec![f64::NEG_INFINITY; s_len];

            // Initial step t = 0
            let lp0_blank = if time_major {
                log_probs.get_3d(0, b, l_prime[0])
            } else {
                log_probs.get_3d(b, 0, l_prime[0])
            };
            alpha[0] = lp0_blank;

            if s_len > 1 {
                let lp0_first = if time_major {
                    log_probs.get_3d(0, b, l_prime[1])
                } else {
                    log_probs.get_3d(b, 0, l_prime[1])
                };
                alpha[1] = lp0_first;
            }

            // Dynamic programming forward pass
            for t in 1..t_len {
                let mut next_alpha = vec![f64::NEG_INFINITY; s_len];
                let start_s = (s_len.saturating_sub(2 * (t_len - t))).max(0);
                let end_s = (2 * (t + 1)).min(s_len);

                for s in start_s..end_s {
                    let label = l_prime[s];
                    if label >= num_classes {
                        continue;
                    }
                    let lp = if time_major {
                        log_probs.get_3d(t, b, label)
                    } else {
                        log_probs.get_3d(b, t, label)
                    };

                    let mut a = alpha[s];
                    if s > 0 {
                        a = log_add(a, alpha[s - 1]);
                    }
                    if s > 1 && l_prime[s] != self.config.blank && l_prime[s] != l_prime[s - 2] {
                        a = log_add(a, alpha[s - 2]);
                    }
                    next_alpha[s] = a + lp;
                }
                alpha = next_alpha;
            }

            // Total probability log P(L|X) = logsumexp(alpha[T-1, S-1], alpha[T-1, S-2])
            let total_log_p = if s_len > 1 {
                log_add(alpha[s_len - 1], alpha[s_len - 2])
            } else {
                alpha[0]
            };

            let loss = if total_log_p.is_infinite() && total_log_p < 0.0 {
                if self.config.zero_infinity {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                -total_log_p
            };

            sample_losses.push(loss);
        }

        Ok(reduction_apply(&sample_losses, self.config.reduction))
    }
}

fn log_add(a: f64, b: f64) -> f64 {
    if a.is_infinite() && a < 0.0 {
        return b;
    }
    if b.is_infinite() && b < 0.0 {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ctc_loss_basic() {
        let ctc = CTCLoss::default();
        // [T=2, N=1, C=3]
        let mut log_probs = Tensor::zeros(vec![2, 1, 3]);
        // Set log probs
        log_probs.set_3d(0, 0, 1, -0.1); // prob approx 0.9 for class 1 at t=0
        log_probs.set_3d(1, 0, 2, -0.1); // prob approx 0.9 for class 2 at t=1

        let targets = vec![vec![1, 2]];
        let loss = ctc.forward(&log_probs, &targets, true).unwrap();
        assert!(loss.item() >= 0.0);
    }
}
