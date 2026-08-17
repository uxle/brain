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

    #[test]
    fn test_prune_schedule_stress_001() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_002() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_003() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_004() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_005() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_006() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_007() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_008() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_009() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_010() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_011() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_012() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_013() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_014() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_015() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_016() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_017() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_018() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_019() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_020() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_021() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_022() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_023() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_024() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_025() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_026() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_027() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_028() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_029() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_030() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_031() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_032() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_033() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_034() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_035() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_036() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_037() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_038() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_039() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_040() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_041() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_042() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_043() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_044() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_045() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_046() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_047() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_048() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_049() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_050() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_051() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_052() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_053() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_054() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_055() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_056() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_057() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_058() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_059() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_060() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_061() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_062() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_063() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_064() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_065() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_066() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_067() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_068() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_069() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_070() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_071() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_072() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_073() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_074() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_075() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_076() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_077() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_078() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_079() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_080() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_081() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_082() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_083() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_084() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_085() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_086() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_087() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_088() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_089() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_090() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_091() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_092() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_093() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_094() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_095() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_096() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_097() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_098() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_099() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_100() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_101() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_102() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_103() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_104() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_105() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_106() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_107() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_108() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_109() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_110() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_111() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_112() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_113() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_114() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_115() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_116() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_117() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_118() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_119() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_120() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_121() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_122() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_123() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_124() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_125() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_126() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_127() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_128() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_129() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_130() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_131() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_132() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_133() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_134() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_135() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_136() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_137() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_138() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_139() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_140() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_141() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_142() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_143() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_144() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_145() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_146() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_147() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_148() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_149() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_150() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_151() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_152() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_153() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_154() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_155() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_156() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_157() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_158() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_159() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_160() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_161() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_162() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_163() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_164() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_165() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_166() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_167() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_168() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_169() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_170() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_171() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_172() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_173() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_174() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_175() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_176() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_177() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_178() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_179() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_180() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_181() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_182() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_183() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_184() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_185() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_186() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_187() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_188() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_189() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_190() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_191() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_192() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_193() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_194() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_195() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_196() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_197() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_198() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_199() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_200() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_201() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_202() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }

    #[test]
    fn test_prune_schedule_stress_203() {
        let sched = IterativePruneSchedule::new(0.0, 0.9, 0, 100, 10);
        let s0 = sched.calculate_sparsity(0);
        let s50 = sched.calculate_sparsity(50);
        let s100 = sched.calculate_sparsity(100);
        assert_eq!(s0, 0.0);
        assert!(s50 > 0.0 && s50 < 0.9);
        assert_eq!(s100, 0.9);
        assert!(sched.should_prune(20));

        let mut lth = LotteryTicketSchedule::new(5, 0.2, 5);
        lth.advance_round();
        assert!(lth.current_cumulative_sparsity() > 0.0);
    }
}
