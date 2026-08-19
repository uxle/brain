//! # Pruning Schedules
//!
//! Iterative magnitude pruning, lottery ticket hypothesis rewind schedules, and polynomial decay curves.
#![allow(missing_docs, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

/// Iterative magnitude pruning schedule gradually increasing target sparsity across epochs.
#[derive(Debug, Clone, PartialEq)]
pub struct IterativePruneSchedule {
    pub initial_sparsity: f64,
    pub final_sparsity: f64,
    pub begin_step: usize,
    pub end_step: usize,
    pub frequency: usize,
}

impl Default for IterativePruneSchedule {
    fn default() -> Self {
        Self {
            initial_sparsity: 0.0,
            final_sparsity: 0.8,
            begin_step: 0,
            end_step: 100,
            frequency: 10,
        }
    }
}

impl IterativePruneSchedule {
    pub fn new(initial_sparsity: f64, final_sparsity: f64, begin_step: usize, end_step: usize, frequency: usize) -> Self {
        Self {
            initial_sparsity,
            final_sparsity,
            begin_step,
            end_step,
            frequency: frequency.max(1),
        }
    }

    /// Computes target sparsity at training step t using cubic polynomial decay curve.
    pub fn calculate_sparsity(&self, step: usize) -> f64 {
        if step < self.begin_step {
            return self.initial_sparsity;
        }
        if step >= self.end_step {
            return self.final_sparsity;
        }

        let progress = (step - self.begin_step) as f64 / (self.end_step - self.begin_step) as f64;
        let factor = 1.0 - (1.0 - progress).powi(3);

        self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * factor
    }

    /// Checks if a pruning step should execute at step t.
    pub fn should_prune(&self, step: usize) -> bool {
        step >= self.begin_step && step <= self.end_step && (step - self.begin_step) % self.frequency == 0
    }
}

/// Lottery Ticket Hypothesis rewinding schedule.
#[derive(Debug, Clone)]
pub struct LotteryTicketSchedule {
    pub rewind_epoch: usize,
    pub prune_ratio_per_round: f64,
    pub num_rounds: usize,
    pub current_round: usize,
}

impl LotteryTicketSchedule {
    pub fn new(rewind_epoch: usize, prune_ratio_per_round: f64, num_rounds: usize) -> Self {
        Self {
            rewind_epoch,
            prune_ratio_per_round,
            num_rounds,
            current_round: 0,
        }
    }

    /// Cumulative sparsity achieved after current round.
    pub fn current_cumulative_sparsity(&self) -> f64 {
        1.0 - (1.0 - self.prune_ratio_per_round).powi(self.current_round as i32)
    }

    /// Advances to next lottery ticket pruning round.
    pub fn advance_round(&mut self) -> bool {
        if self.current_round < self.num_rounds {
            self.current_round += 1;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
