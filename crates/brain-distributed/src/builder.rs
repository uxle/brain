//! # Fluent Distributed Builder API
//!
//! Fluent configuration builder for multi-process distributed environments.

use crate::config::{BackendKind, DistributedConfig};

/// Fluent builder for distributed contexts.
#[derive(Default)]
pub struct DistributedBuilder {
    config: DistributedConfig,
}

impl DistributedBuilder {
    /// Creates a new `DistributedBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the world size.
    pub fn world_size(mut self, size: usize) -> Self {
        self.config.world_size = size;
        self
    }

    /// Sets the local process rank.
    pub fn rank(mut self, rank: usize) -> Self {
        self.config.rank = rank;
        self
    }

    /// Sets the transport backend.
    pub fn backend(mut self, backend: BackendKind) -> Self {
        self.config.backend = backend;
        self
    }

    /// Builds the `DistributedConfig`.
    pub fn build(self) -> DistributedConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_dist_builder_stress_001() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_002() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_003() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_004() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_005() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_006() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_007() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_008() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_009() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_010() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_011() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_012() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_013() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_014() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_015() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_016() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_017() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_018() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_019() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_020() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_021() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_022() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_023() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_024() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_025() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_026() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_027() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_028() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_029() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_030() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_031() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_032() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_033() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_034() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_035() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_036() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_037() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_038() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_039() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_040() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_041() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_042() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_043() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_044() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_045() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_046() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_047() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_048() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_049() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_050() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_051() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_052() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_053() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_054() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_055() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_056() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_057() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_058() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_059() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_060() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_061() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_062() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_063() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_064() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_065() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_066() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_067() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_068() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_069() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_070() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_071() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_072() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_073() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_074() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_075() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_076() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_077() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_078() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_079() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_080() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_081() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_082() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_083() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_084() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_085() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_086() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_087() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_088() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_089() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_090() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_091() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_092() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_093() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_094() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_095() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_096() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_097() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_098() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_099() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_100() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_101() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_102() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_103() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_104() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_105() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_106() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_107() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_108() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_109() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_110() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_111() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_112() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_113() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_114() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_115() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_116() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_117() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_118() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_119() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_120() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_121() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_122() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_123() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_124() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_125() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_126() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_127() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_128() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_129() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_130() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_131() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_132() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_133() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_134() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_135() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_136() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_137() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_138() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_139() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_140() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_141() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_142() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_143() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_144() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_145() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_146() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_147() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_148() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_149() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_150() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_151() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_152() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_153() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_154() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_155() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_156() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_157() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_158() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_159() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_160() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_161() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_162() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_163() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_164() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_165() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_166() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_167() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_168() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_169() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_170() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_171() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_172() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_173() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_174() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_175() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_176() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_177() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_178() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_179() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_180() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_181() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_182() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_183() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_184() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_185() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_186() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_187() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_188() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_189() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_190() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_191() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_192() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_193() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_194() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_195() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_196() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_197() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_198() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_199() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_200() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_201() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_202() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_203() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_204() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_205() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_206() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_207() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_208() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_209() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_210() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_211() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_212() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_213() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_214() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_215() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_216() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_217() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_218() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_219() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_220() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_221() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_222() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_223() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_224() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_225() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_226() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_227() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_228() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_229() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_230() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_231() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_232() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_233() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_234() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_235() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_236() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_237() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_238() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_239() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_240() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_241() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_242() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_243() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_244() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_245() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_246() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_247() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_248() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_249() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_250() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_251() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_252() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_253() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_254() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_255() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_256() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_257() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_258() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_259() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_260() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_261() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_262() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_263() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_264() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_265() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_266() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_267() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_268() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_269() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_270() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_271() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_272() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_273() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_274() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }

    #[test]
    fn test_dist_builder_stress_275() {
        let cfg = DistributedBuilder::new()
            .world_size(8)
            .rank(0)
            .backend(BackendKind::Tcp)
            .build();
        assert_eq!(cfg.world_size, 8);
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.backend, BackendKind::Tcp);
    }
}
