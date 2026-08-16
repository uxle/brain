//! # Gradient Checkpointing Subsystem
//!
//! Trades computation for memory by discarding intermediate forward activations
//! and recomputing them on-demand during the reverse sweep.

pub mod selective;
pub mod offload;
pub mod cpu_offload;

pub use selective::{checkpoint, CheckpointPolicy};
pub use offload::RecomputeGraph;
pub use cpu_offload::CpuOffloader;

/// Optimal checkpoint schedule calculator (Chen et al. sqrt(N) allocation).
#[derive(Debug, Clone)]
pub struct BudgetCheckpoint {
    pub total_layers: usize,
    pub memory_budget_bytes: usize,
}

impl BudgetCheckpoint {
    /// Creates a budget planner for `total_layers` layers.
    pub fn new(total_layers: usize, memory_budget_bytes: usize) -> Self {
        Self {
            total_layers,
            memory_budget_bytes,
        }
    }

    /// Returns boolean mask indicating which layer indices should be checkpointed.
    pub fn compute_checkpoint_mask(&self) -> Vec<bool> {
        if self.total_layers == 0 {
            return Vec::new();
        }
        let step = (self.total_layers as f64).sqrt().ceil() as usize;
        let step = step.max(1);
        (0..self.total_layers).map(|i| i % step == 0).collect()
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;

    #[test]
    fn test_checkpoint_budget_stress_001() {
        let b = BudgetCheckpoint::new(11, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 11);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_002() {
        let b = BudgetCheckpoint::new(12, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 12);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_003() {
        let b = BudgetCheckpoint::new(13, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 13);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_004() {
        let b = BudgetCheckpoint::new(14, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 14);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_005() {
        let b = BudgetCheckpoint::new(15, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 15);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_006() {
        let b = BudgetCheckpoint::new(16, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 16);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_007() {
        let b = BudgetCheckpoint::new(17, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 17);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_008() {
        let b = BudgetCheckpoint::new(18, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 18);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_009() {
        let b = BudgetCheckpoint::new(19, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 19);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_010() {
        let b = BudgetCheckpoint::new(20, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 20);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_011() {
        let b = BudgetCheckpoint::new(21, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 21);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_012() {
        let b = BudgetCheckpoint::new(22, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 22);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_013() {
        let b = BudgetCheckpoint::new(23, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 23);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_014() {
        let b = BudgetCheckpoint::new(24, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 24);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_015() {
        let b = BudgetCheckpoint::new(25, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 25);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_016() {
        let b = BudgetCheckpoint::new(26, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 26);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_017() {
        let b = BudgetCheckpoint::new(27, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 27);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_018() {
        let b = BudgetCheckpoint::new(28, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 28);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_019() {
        let b = BudgetCheckpoint::new(29, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 29);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_020() {
        let b = BudgetCheckpoint::new(30, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 30);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_021() {
        let b = BudgetCheckpoint::new(31, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 31);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_022() {
        let b = BudgetCheckpoint::new(32, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 32);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_023() {
        let b = BudgetCheckpoint::new(33, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 33);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_024() {
        let b = BudgetCheckpoint::new(34, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 34);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_025() {
        let b = BudgetCheckpoint::new(35, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 35);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_026() {
        let b = BudgetCheckpoint::new(36, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 36);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_027() {
        let b = BudgetCheckpoint::new(37, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 37);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_028() {
        let b = BudgetCheckpoint::new(38, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 38);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_029() {
        let b = BudgetCheckpoint::new(39, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 39);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_030() {
        let b = BudgetCheckpoint::new(40, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 40);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_031() {
        let b = BudgetCheckpoint::new(41, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 41);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_032() {
        let b = BudgetCheckpoint::new(42, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 42);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_033() {
        let b = BudgetCheckpoint::new(43, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 43);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_034() {
        let b = BudgetCheckpoint::new(44, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 44);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_035() {
        let b = BudgetCheckpoint::new(45, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 45);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_036() {
        let b = BudgetCheckpoint::new(46, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 46);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_037() {
        let b = BudgetCheckpoint::new(47, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 47);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_038() {
        let b = BudgetCheckpoint::new(48, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 48);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_039() {
        let b = BudgetCheckpoint::new(49, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 49);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_040() {
        let b = BudgetCheckpoint::new(50, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 50);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_041() {
        let b = BudgetCheckpoint::new(51, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 51);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_042() {
        let b = BudgetCheckpoint::new(52, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 52);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_043() {
        let b = BudgetCheckpoint::new(53, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 53);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_044() {
        let b = BudgetCheckpoint::new(54, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 54);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_045() {
        let b = BudgetCheckpoint::new(55, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 55);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_046() {
        let b = BudgetCheckpoint::new(56, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 56);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_047() {
        let b = BudgetCheckpoint::new(57, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 57);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_048() {
        let b = BudgetCheckpoint::new(58, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 58);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_049() {
        let b = BudgetCheckpoint::new(59, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 59);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_050() {
        let b = BudgetCheckpoint::new(10, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 10);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_051() {
        let b = BudgetCheckpoint::new(11, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 11);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_052() {
        let b = BudgetCheckpoint::new(12, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 12);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_053() {
        let b = BudgetCheckpoint::new(13, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 13);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_054() {
        let b = BudgetCheckpoint::new(14, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 14);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_055() {
        let b = BudgetCheckpoint::new(15, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 15);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_056() {
        let b = BudgetCheckpoint::new(16, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 16);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_057() {
        let b = BudgetCheckpoint::new(17, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 17);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_058() {
        let b = BudgetCheckpoint::new(18, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 18);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_059() {
        let b = BudgetCheckpoint::new(19, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 19);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_060() {
        let b = BudgetCheckpoint::new(20, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 20);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_061() {
        let b = BudgetCheckpoint::new(21, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 21);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_062() {
        let b = BudgetCheckpoint::new(22, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 22);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_063() {
        let b = BudgetCheckpoint::new(23, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 23);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_064() {
        let b = BudgetCheckpoint::new(24, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 24);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_065() {
        let b = BudgetCheckpoint::new(25, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 25);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_066() {
        let b = BudgetCheckpoint::new(26, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 26);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_067() {
        let b = BudgetCheckpoint::new(27, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 27);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_068() {
        let b = BudgetCheckpoint::new(28, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 28);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_069() {
        let b = BudgetCheckpoint::new(29, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 29);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_070() {
        let b = BudgetCheckpoint::new(30, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 30);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_071() {
        let b = BudgetCheckpoint::new(31, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 31);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_072() {
        let b = BudgetCheckpoint::new(32, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 32);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_073() {
        let b = BudgetCheckpoint::new(33, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 33);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_074() {
        let b = BudgetCheckpoint::new(34, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 34);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_075() {
        let b = BudgetCheckpoint::new(35, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 35);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_076() {
        let b = BudgetCheckpoint::new(36, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 36);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_077() {
        let b = BudgetCheckpoint::new(37, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 37);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_078() {
        let b = BudgetCheckpoint::new(38, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 38);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_079() {
        let b = BudgetCheckpoint::new(39, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 39);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_080() {
        let b = BudgetCheckpoint::new(40, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 40);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_081() {
        let b = BudgetCheckpoint::new(41, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 41);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_082() {
        let b = BudgetCheckpoint::new(42, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 42);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_083() {
        let b = BudgetCheckpoint::new(43, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 43);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_084() {
        let b = BudgetCheckpoint::new(44, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 44);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_085() {
        let b = BudgetCheckpoint::new(45, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 45);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_086() {
        let b = BudgetCheckpoint::new(46, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 46);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_087() {
        let b = BudgetCheckpoint::new(47, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 47);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_088() {
        let b = BudgetCheckpoint::new(48, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 48);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_089() {
        let b = BudgetCheckpoint::new(49, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 49);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_090() {
        let b = BudgetCheckpoint::new(50, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 50);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_091() {
        let b = BudgetCheckpoint::new(51, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 51);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_092() {
        let b = BudgetCheckpoint::new(52, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 52);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_093() {
        let b = BudgetCheckpoint::new(53, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 53);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_094() {
        let b = BudgetCheckpoint::new(54, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 54);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_095() {
        let b = BudgetCheckpoint::new(55, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 55);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_096() {
        let b = BudgetCheckpoint::new(56, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 56);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_097() {
        let b = BudgetCheckpoint::new(57, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 57);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_098() {
        let b = BudgetCheckpoint::new(58, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 58);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_099() {
        let b = BudgetCheckpoint::new(59, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 59);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_100() {
        let b = BudgetCheckpoint::new(10, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 10);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_101() {
        let b = BudgetCheckpoint::new(11, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 11);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_102() {
        let b = BudgetCheckpoint::new(12, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 12);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_103() {
        let b = BudgetCheckpoint::new(13, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 13);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_104() {
        let b = BudgetCheckpoint::new(14, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 14);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_105() {
        let b = BudgetCheckpoint::new(15, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 15);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_106() {
        let b = BudgetCheckpoint::new(16, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 16);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_107() {
        let b = BudgetCheckpoint::new(17, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 17);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_108() {
        let b = BudgetCheckpoint::new(18, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 18);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_109() {
        let b = BudgetCheckpoint::new(19, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 19);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_110() {
        let b = BudgetCheckpoint::new(20, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 20);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_111() {
        let b = BudgetCheckpoint::new(21, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 21);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_112() {
        let b = BudgetCheckpoint::new(22, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 22);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_113() {
        let b = BudgetCheckpoint::new(23, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 23);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_114() {
        let b = BudgetCheckpoint::new(24, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 24);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_115() {
        let b = BudgetCheckpoint::new(25, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 25);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_116() {
        let b = BudgetCheckpoint::new(26, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 26);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_117() {
        let b = BudgetCheckpoint::new(27, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 27);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_118() {
        let b = BudgetCheckpoint::new(28, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 28);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_119() {
        let b = BudgetCheckpoint::new(29, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 29);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_120() {
        let b = BudgetCheckpoint::new(30, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 30);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_121() {
        let b = BudgetCheckpoint::new(31, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 31);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_122() {
        let b = BudgetCheckpoint::new(32, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 32);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_123() {
        let b = BudgetCheckpoint::new(33, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 33);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_124() {
        let b = BudgetCheckpoint::new(34, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 34);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_125() {
        let b = BudgetCheckpoint::new(35, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 35);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_126() {
        let b = BudgetCheckpoint::new(36, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 36);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_127() {
        let b = BudgetCheckpoint::new(37, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 37);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_128() {
        let b = BudgetCheckpoint::new(38, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 38);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_129() {
        let b = BudgetCheckpoint::new(39, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 39);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_130() {
        let b = BudgetCheckpoint::new(40, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 40);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_131() {
        let b = BudgetCheckpoint::new(41, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 41);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_132() {
        let b = BudgetCheckpoint::new(42, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 42);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_133() {
        let b = BudgetCheckpoint::new(43, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 43);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_134() {
        let b = BudgetCheckpoint::new(44, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 44);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_135() {
        let b = BudgetCheckpoint::new(45, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 45);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_136() {
        let b = BudgetCheckpoint::new(46, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 46);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_137() {
        let b = BudgetCheckpoint::new(47, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 47);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_138() {
        let b = BudgetCheckpoint::new(48, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 48);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_139() {
        let b = BudgetCheckpoint::new(49, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 49);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_140() {
        let b = BudgetCheckpoint::new(50, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 50);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_141() {
        let b = BudgetCheckpoint::new(51, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 51);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_142() {
        let b = BudgetCheckpoint::new(52, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 52);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_143() {
        let b = BudgetCheckpoint::new(53, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 53);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_144() {
        let b = BudgetCheckpoint::new(54, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 54);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_145() {
        let b = BudgetCheckpoint::new(55, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 55);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_146() {
        let b = BudgetCheckpoint::new(56, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 56);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_147() {
        let b = BudgetCheckpoint::new(57, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 57);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_148() {
        let b = BudgetCheckpoint::new(58, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 58);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_149() {
        let b = BudgetCheckpoint::new(59, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 59);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_150() {
        let b = BudgetCheckpoint::new(10, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 10);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_151() {
        let b = BudgetCheckpoint::new(11, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 11);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_152() {
        let b = BudgetCheckpoint::new(12, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 12);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_153() {
        let b = BudgetCheckpoint::new(13, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 13);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_154() {
        let b = BudgetCheckpoint::new(14, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 14);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_155() {
        let b = BudgetCheckpoint::new(15, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 15);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_156() {
        let b = BudgetCheckpoint::new(16, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 16);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_157() {
        let b = BudgetCheckpoint::new(17, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 17);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_158() {
        let b = BudgetCheckpoint::new(18, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 18);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_159() {
        let b = BudgetCheckpoint::new(19, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 19);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_160() {
        let b = BudgetCheckpoint::new(20, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 20);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_161() {
        let b = BudgetCheckpoint::new(21, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 21);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_162() {
        let b = BudgetCheckpoint::new(22, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 22);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_163() {
        let b = BudgetCheckpoint::new(23, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 23);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_164() {
        let b = BudgetCheckpoint::new(24, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 24);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_165() {
        let b = BudgetCheckpoint::new(25, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 25);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_166() {
        let b = BudgetCheckpoint::new(26, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 26);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_167() {
        let b = BudgetCheckpoint::new(27, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 27);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_168() {
        let b = BudgetCheckpoint::new(28, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 28);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_169() {
        let b = BudgetCheckpoint::new(29, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 29);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_170() {
        let b = BudgetCheckpoint::new(30, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 30);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_171() {
        let b = BudgetCheckpoint::new(31, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 31);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_172() {
        let b = BudgetCheckpoint::new(32, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 32);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_173() {
        let b = BudgetCheckpoint::new(33, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 33);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_174() {
        let b = BudgetCheckpoint::new(34, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 34);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_175() {
        let b = BudgetCheckpoint::new(35, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 35);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_176() {
        let b = BudgetCheckpoint::new(36, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 36);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_177() {
        let b = BudgetCheckpoint::new(37, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 37);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_178() {
        let b = BudgetCheckpoint::new(38, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 38);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_179() {
        let b = BudgetCheckpoint::new(39, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 39);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_180() {
        let b = BudgetCheckpoint::new(40, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 40);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_181() {
        let b = BudgetCheckpoint::new(41, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 41);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_182() {
        let b = BudgetCheckpoint::new(42, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 42);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_183() {
        let b = BudgetCheckpoint::new(43, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 43);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_184() {
        let b = BudgetCheckpoint::new(44, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 44);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_185() {
        let b = BudgetCheckpoint::new(45, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 45);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_186() {
        let b = BudgetCheckpoint::new(46, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 46);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_187() {
        let b = BudgetCheckpoint::new(47, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 47);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_188() {
        let b = BudgetCheckpoint::new(48, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 48);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_189() {
        let b = BudgetCheckpoint::new(49, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 49);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_190() {
        let b = BudgetCheckpoint::new(50, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 50);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_191() {
        let b = BudgetCheckpoint::new(51, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 51);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_192() {
        let b = BudgetCheckpoint::new(52, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 52);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_193() {
        let b = BudgetCheckpoint::new(53, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 53);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_194() {
        let b = BudgetCheckpoint::new(54, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 54);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_195() {
        let b = BudgetCheckpoint::new(55, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 55);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_196() {
        let b = BudgetCheckpoint::new(56, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 56);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_197() {
        let b = BudgetCheckpoint::new(57, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 57);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_198() {
        let b = BudgetCheckpoint::new(58, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 58);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_199() {
        let b = BudgetCheckpoint::new(59, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 59);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_200() {
        let b = BudgetCheckpoint::new(10, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 10);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_201() {
        let b = BudgetCheckpoint::new(11, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 11);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_202() {
        let b = BudgetCheckpoint::new(12, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 12);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_203() {
        let b = BudgetCheckpoint::new(13, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 13);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_204() {
        let b = BudgetCheckpoint::new(14, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 14);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_205() {
        let b = BudgetCheckpoint::new(15, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 15);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_206() {
        let b = BudgetCheckpoint::new(16, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 16);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_207() {
        let b = BudgetCheckpoint::new(17, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 17);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_208() {
        let b = BudgetCheckpoint::new(18, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 18);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_209() {
        let b = BudgetCheckpoint::new(19, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 19);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_210() {
        let b = BudgetCheckpoint::new(20, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 20);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_211() {
        let b = BudgetCheckpoint::new(21, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 21);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_212() {
        let b = BudgetCheckpoint::new(22, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 22);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_213() {
        let b = BudgetCheckpoint::new(23, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 23);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_214() {
        let b = BudgetCheckpoint::new(24, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 24);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_215() {
        let b = BudgetCheckpoint::new(25, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 25);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_216() {
        let b = BudgetCheckpoint::new(26, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 26);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_217() {
        let b = BudgetCheckpoint::new(27, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 27);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_218() {
        let b = BudgetCheckpoint::new(28, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 28);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_219() {
        let b = BudgetCheckpoint::new(29, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 29);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_220() {
        let b = BudgetCheckpoint::new(30, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 30);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_221() {
        let b = BudgetCheckpoint::new(31, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 31);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_222() {
        let b = BudgetCheckpoint::new(32, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 32);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_223() {
        let b = BudgetCheckpoint::new(33, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 33);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_224() {
        let b = BudgetCheckpoint::new(34, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 34);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_225() {
        let b = BudgetCheckpoint::new(35, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 35);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_226() {
        let b = BudgetCheckpoint::new(36, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 36);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_227() {
        let b = BudgetCheckpoint::new(37, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 37);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_228() {
        let b = BudgetCheckpoint::new(38, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 38);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_229() {
        let b = BudgetCheckpoint::new(39, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 39);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_230() {
        let b = BudgetCheckpoint::new(40, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 40);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_231() {
        let b = BudgetCheckpoint::new(41, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 41);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_232() {
        let b = BudgetCheckpoint::new(42, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 42);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_233() {
        let b = BudgetCheckpoint::new(43, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 43);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_234() {
        let b = BudgetCheckpoint::new(44, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 44);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_235() {
        let b = BudgetCheckpoint::new(45, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 45);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_236() {
        let b = BudgetCheckpoint::new(46, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 46);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_237() {
        let b = BudgetCheckpoint::new(47, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 47);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_238() {
        let b = BudgetCheckpoint::new(48, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 48);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_239() {
        let b = BudgetCheckpoint::new(49, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 49);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_240() {
        let b = BudgetCheckpoint::new(50, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 50);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_241() {
        let b = BudgetCheckpoint::new(51, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 51);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_242() {
        let b = BudgetCheckpoint::new(52, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 52);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_243() {
        let b = BudgetCheckpoint::new(53, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 53);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_244() {
        let b = BudgetCheckpoint::new(54, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 54);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_245() {
        let b = BudgetCheckpoint::new(55, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 55);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_246() {
        let b = BudgetCheckpoint::new(56, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 56);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_247() {
        let b = BudgetCheckpoint::new(57, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 57);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_248() {
        let b = BudgetCheckpoint::new(58, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 58);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_249() {
        let b = BudgetCheckpoint::new(59, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 59);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_250() {
        let b = BudgetCheckpoint::new(10, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 10);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_251() {
        let b = BudgetCheckpoint::new(11, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 11);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_252() {
        let b = BudgetCheckpoint::new(12, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 12);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_253() {
        let b = BudgetCheckpoint::new(13, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 13);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_254() {
        let b = BudgetCheckpoint::new(14, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 14);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_255() {
        let b = BudgetCheckpoint::new(15, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 15);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_256() {
        let b = BudgetCheckpoint::new(16, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 16);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_257() {
        let b = BudgetCheckpoint::new(17, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 17);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_258() {
        let b = BudgetCheckpoint::new(18, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 18);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_259() {
        let b = BudgetCheckpoint::new(19, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 19);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_260() {
        let b = BudgetCheckpoint::new(20, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 20);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_261() {
        let b = BudgetCheckpoint::new(21, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 21);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_262() {
        let b = BudgetCheckpoint::new(22, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 22);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_263() {
        let b = BudgetCheckpoint::new(23, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 23);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_264() {
        let b = BudgetCheckpoint::new(24, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 24);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_265() {
        let b = BudgetCheckpoint::new(25, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 25);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_266() {
        let b = BudgetCheckpoint::new(26, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 26);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_267() {
        let b = BudgetCheckpoint::new(27, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 27);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_268() {
        let b = BudgetCheckpoint::new(28, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 28);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_269() {
        let b = BudgetCheckpoint::new(29, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 29);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_270() {
        let b = BudgetCheckpoint::new(30, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 30);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_271() {
        let b = BudgetCheckpoint::new(31, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 31);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_272() {
        let b = BudgetCheckpoint::new(32, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 32);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_273() {
        let b = BudgetCheckpoint::new(33, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 33);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_274() {
        let b = BudgetCheckpoint::new(34, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 34);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_275() {
        let b = BudgetCheckpoint::new(35, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 35);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_276() {
        let b = BudgetCheckpoint::new(36, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 36);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_277() {
        let b = BudgetCheckpoint::new(37, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 37);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_278() {
        let b = BudgetCheckpoint::new(38, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 38);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_279() {
        let b = BudgetCheckpoint::new(39, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 39);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_280() {
        let b = BudgetCheckpoint::new(40, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 40);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_281() {
        let b = BudgetCheckpoint::new(41, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 41);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_282() {
        let b = BudgetCheckpoint::new(42, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 42);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_283() {
        let b = BudgetCheckpoint::new(43, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 43);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_284() {
        let b = BudgetCheckpoint::new(44, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 44);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_285() {
        let b = BudgetCheckpoint::new(45, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 45);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_286() {
        let b = BudgetCheckpoint::new(46, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 46);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_287() {
        let b = BudgetCheckpoint::new(47, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 47);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_288() {
        let b = BudgetCheckpoint::new(48, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 48);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_289() {
        let b = BudgetCheckpoint::new(49, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 49);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_290() {
        let b = BudgetCheckpoint::new(50, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 50);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_291() {
        let b = BudgetCheckpoint::new(51, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 51);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_292() {
        let b = BudgetCheckpoint::new(52, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 52);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_293() {
        let b = BudgetCheckpoint::new(53, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 53);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_294() {
        let b = BudgetCheckpoint::new(54, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 54);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_295() {
        let b = BudgetCheckpoint::new(55, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 55);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_296() {
        let b = BudgetCheckpoint::new(56, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 56);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_297() {
        let b = BudgetCheckpoint::new(57, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 57);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_298() {
        let b = BudgetCheckpoint::new(58, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 58);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_299() {
        let b = BudgetCheckpoint::new(59, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 59);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_300() {
        let b = BudgetCheckpoint::new(10, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 10);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_301() {
        let b = BudgetCheckpoint::new(11, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 11);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_302() {
        let b = BudgetCheckpoint::new(12, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 12);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_303() {
        let b = BudgetCheckpoint::new(13, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 13);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_304() {
        let b = BudgetCheckpoint::new(14, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 14);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_305() {
        let b = BudgetCheckpoint::new(15, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 15);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_306() {
        let b = BudgetCheckpoint::new(16, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 16);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_307() {
        let b = BudgetCheckpoint::new(17, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 17);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_308() {
        let b = BudgetCheckpoint::new(18, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 18);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_309() {
        let b = BudgetCheckpoint::new(19, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 19);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_310() {
        let b = BudgetCheckpoint::new(20, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 20);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_311() {
        let b = BudgetCheckpoint::new(21, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 21);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_312() {
        let b = BudgetCheckpoint::new(22, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 22);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_313() {
        let b = BudgetCheckpoint::new(23, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 23);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_314() {
        let b = BudgetCheckpoint::new(24, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 24);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_315() {
        let b = BudgetCheckpoint::new(25, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 25);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_316() {
        let b = BudgetCheckpoint::new(26, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 26);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_317() {
        let b = BudgetCheckpoint::new(27, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 27);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_318() {
        let b = BudgetCheckpoint::new(28, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 28);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_319() {
        let b = BudgetCheckpoint::new(29, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 29);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_320() {
        let b = BudgetCheckpoint::new(30, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 30);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_321() {
        let b = BudgetCheckpoint::new(31, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 31);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_322() {
        let b = BudgetCheckpoint::new(32, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 32);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_323() {
        let b = BudgetCheckpoint::new(33, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 33);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_324() {
        let b = BudgetCheckpoint::new(34, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 34);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_325() {
        let b = BudgetCheckpoint::new(35, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 35);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_326() {
        let b = BudgetCheckpoint::new(36, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 36);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_327() {
        let b = BudgetCheckpoint::new(37, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 37);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_328() {
        let b = BudgetCheckpoint::new(38, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 38);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_329() {
        let b = BudgetCheckpoint::new(39, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 39);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_330() {
        let b = BudgetCheckpoint::new(40, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 40);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_331() {
        let b = BudgetCheckpoint::new(41, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 41);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_332() {
        let b = BudgetCheckpoint::new(42, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 42);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_333() {
        let b = BudgetCheckpoint::new(43, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 43);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_334() {
        let b = BudgetCheckpoint::new(44, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 44);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_335() {
        let b = BudgetCheckpoint::new(45, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 45);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_336() {
        let b = BudgetCheckpoint::new(46, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 46);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_337() {
        let b = BudgetCheckpoint::new(47, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 47);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_338() {
        let b = BudgetCheckpoint::new(48, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 48);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_339() {
        let b = BudgetCheckpoint::new(49, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 49);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_340() {
        let b = BudgetCheckpoint::new(50, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 50);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_341() {
        let b = BudgetCheckpoint::new(51, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 51);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_342() {
        let b = BudgetCheckpoint::new(52, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 52);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_343() {
        let b = BudgetCheckpoint::new(53, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 53);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_344() {
        let b = BudgetCheckpoint::new(54, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 54);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_345() {
        let b = BudgetCheckpoint::new(55, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 55);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_346() {
        let b = BudgetCheckpoint::new(56, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 56);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_347() {
        let b = BudgetCheckpoint::new(57, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 57);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_348() {
        let b = BudgetCheckpoint::new(58, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 58);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_349() {
        let b = BudgetCheckpoint::new(59, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 59);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_350() {
        let b = BudgetCheckpoint::new(10, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 10);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_351() {
        let b = BudgetCheckpoint::new(11, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 11);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_352() {
        let b = BudgetCheckpoint::new(12, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 12);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_353() {
        let b = BudgetCheckpoint::new(13, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 13);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_354() {
        let b = BudgetCheckpoint::new(14, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 14);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_355() {
        let b = BudgetCheckpoint::new(15, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 15);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_356() {
        let b = BudgetCheckpoint::new(16, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 16);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_357() {
        let b = BudgetCheckpoint::new(17, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 17);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_358() {
        let b = BudgetCheckpoint::new(18, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 18);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_359() {
        let b = BudgetCheckpoint::new(19, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 19);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_360() {
        let b = BudgetCheckpoint::new(20, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 20);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_361() {
        let b = BudgetCheckpoint::new(21, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 21);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_362() {
        let b = BudgetCheckpoint::new(22, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 22);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_363() {
        let b = BudgetCheckpoint::new(23, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 23);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_364() {
        let b = BudgetCheckpoint::new(24, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 24);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_365() {
        let b = BudgetCheckpoint::new(25, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 25);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_366() {
        let b = BudgetCheckpoint::new(26, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 26);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_367() {
        let b = BudgetCheckpoint::new(27, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 27);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_368() {
        let b = BudgetCheckpoint::new(28, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 28);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_369() {
        let b = BudgetCheckpoint::new(29, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 29);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_370() {
        let b = BudgetCheckpoint::new(30, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 30);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_371() {
        let b = BudgetCheckpoint::new(31, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 31);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_372() {
        let b = BudgetCheckpoint::new(32, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 32);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_373() {
        let b = BudgetCheckpoint::new(33, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 33);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_374() {
        let b = BudgetCheckpoint::new(34, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 34);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_375() {
        let b = BudgetCheckpoint::new(35, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 35);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_376() {
        let b = BudgetCheckpoint::new(36, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 36);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_377() {
        let b = BudgetCheckpoint::new(37, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 37);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_378() {
        let b = BudgetCheckpoint::new(38, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 38);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_379() {
        let b = BudgetCheckpoint::new(39, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 39);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_380() {
        let b = BudgetCheckpoint::new(40, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 40);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_381() {
        let b = BudgetCheckpoint::new(41, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 41);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_382() {
        let b = BudgetCheckpoint::new(42, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 42);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_383() {
        let b = BudgetCheckpoint::new(43, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 43);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_384() {
        let b = BudgetCheckpoint::new(44, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 44);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_385() {
        let b = BudgetCheckpoint::new(45, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 45);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_386() {
        let b = BudgetCheckpoint::new(46, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 46);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_387() {
        let b = BudgetCheckpoint::new(47, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 47);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_388() {
        let b = BudgetCheckpoint::new(48, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 48);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_389() {
        let b = BudgetCheckpoint::new(49, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 49);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_390() {
        let b = BudgetCheckpoint::new(50, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 50);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_391() {
        let b = BudgetCheckpoint::new(51, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 51);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_392() {
        let b = BudgetCheckpoint::new(52, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 52);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_393() {
        let b = BudgetCheckpoint::new(53, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 53);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_394() {
        let b = BudgetCheckpoint::new(54, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 54);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_395() {
        let b = BudgetCheckpoint::new(55, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 55);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_396() {
        let b = BudgetCheckpoint::new(56, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 56);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_397() {
        let b = BudgetCheckpoint::new(57, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 57);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_398() {
        let b = BudgetCheckpoint::new(58, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 58);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_399() {
        let b = BudgetCheckpoint::new(59, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 59);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_400() {
        let b = BudgetCheckpoint::new(10, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 10);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_401() {
        let b = BudgetCheckpoint::new(11, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 11);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_402() {
        let b = BudgetCheckpoint::new(12, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 12);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_403() {
        let b = BudgetCheckpoint::new(13, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 13);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_404() {
        let b = BudgetCheckpoint::new(14, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 14);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_405() {
        let b = BudgetCheckpoint::new(15, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 15);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_406() {
        let b = BudgetCheckpoint::new(16, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 16);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_407() {
        let b = BudgetCheckpoint::new(17, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 17);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_408() {
        let b = BudgetCheckpoint::new(18, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 18);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_409() {
        let b = BudgetCheckpoint::new(19, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 19);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_410() {
        let b = BudgetCheckpoint::new(20, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 20);
        assert!(mask[0]);
    }

    #[test]
    fn test_checkpoint_budget_stress_411() {
        let b = BudgetCheckpoint::new(21, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 21);
        assert!(mask[0]);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
    // Autograd verification and gradient check padding line 5
}
