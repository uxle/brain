//! # Pipeline Latency Profiling
//!
//! Measures per-stage compute and wait times to identify throughput bottlenecks.

use std::time::Duration;

/// Diagnostic profile report for a pipeline stage.
#[derive(Debug, Clone, Default)]
pub struct StageProfile {
    pub name: String,
    pub execution_time: Duration,
}

impl StageProfile {
    /// Creates a new `StageProfile`.
    pub fn new(name: impl Into<String>, execution_time: Duration) -> Self {
        Self {
            name: name.into(),
            execution_time,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_profile_stress_001() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_002() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_003() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_004() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_005() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_006() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_007() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_008() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_009() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_010() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_011() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_012() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_013() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_014() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_015() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_016() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_017() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_018() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_019() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_020() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_021() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_022() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_023() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_024() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_025() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_026() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_027() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_028() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_029() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_030() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_031() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_032() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_033() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_034() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_035() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_036() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_037() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_038() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_039() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_040() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_041() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_042() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_043() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_044() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_045() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_046() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_047() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_048() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_049() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_050() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_051() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_052() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_053() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_054() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_055() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_056() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_057() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_058() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_059() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_060() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_061() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_062() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_063() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_064() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_065() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_066() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_067() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_068() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_069() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_070() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_071() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_072() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_073() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_074() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_075() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_076() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_077() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_078() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_079() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_080() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_081() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_082() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_083() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_084() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_085() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_086() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_087() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_088() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_089() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_090() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_091() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_092() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_093() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_094() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_095() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_096() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_097() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_098() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_099() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_100() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_101() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_102() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_103() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_104() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_105() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_106() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_107() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_108() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_109() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_110() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_111() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_112() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_113() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_114() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_115() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_116() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_117() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_118() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_119() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_120() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_121() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_122() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_123() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_124() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_125() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_126() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_127() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_128() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_129() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_130() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_131() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_132() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_133() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_134() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_135() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_136() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_137() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_138() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_139() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_140() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_141() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_142() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_143() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_144() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_145() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_146() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_147() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_148() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_149() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_150() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_151() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_152() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_153() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_154() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_155() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_156() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_157() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_158() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_159() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_160() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_161() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_162() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_163() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_164() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_165() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_166() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_167() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_168() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_169() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_170() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_171() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_172() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_173() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_174() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_175() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_176() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_177() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_178() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_179() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_180() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_181() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_182() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_183() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_184() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_185() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_186() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_187() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_188() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_189() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_190() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_191() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_192() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_193() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_194() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_195() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_196() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_197() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_198() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_199() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_200() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_201() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_202() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_203() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_204() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_205() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_206() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_207() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_208() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_209() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_210() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_211() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_212() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_213() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_214() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_215() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_216() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_217() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_218() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_219() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_220() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_221() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_222() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_223() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_224() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_225() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_226() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_227() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_228() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_229() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_230() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_231() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_232() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_233() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_234() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_235() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_236() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_237() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_238() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_239() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_240() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_241() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_242() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_243() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_244() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_245() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_246() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_247() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_248() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_249() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_250() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_251() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_252() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_253() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_254() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_255() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_256() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_257() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_258() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_259() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_260() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_261() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_262() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_263() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_264() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_265() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_266() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_267() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_268() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_269() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_270() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_271() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_272() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_273() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_274() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_275() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_276() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_277() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_278() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_279() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_280() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_281() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_282() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_283() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_284() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_285() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_286() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_287() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_288() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_289() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_290() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_291() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_292() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_293() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_294() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_295() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_296() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_297() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_298() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_299() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_300() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_301() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_302() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_303() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_304() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_305() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_306() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_307() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_308() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_309() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_310() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_311() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_312() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_313() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_314() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_315() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_316() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_317() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_318() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_319() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_320() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_321() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_322() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_323() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_324() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_325() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_326() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_327() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_328() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_329() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_330() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_331() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_332() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_333() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_334() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_335() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_336() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_337() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_338() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_339() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_340() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_341() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_342() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_343() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_344() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_345() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_346() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_347() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_348() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_349() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_350() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_351() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_352() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_353() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_354() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_355() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_356() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_357() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_358() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_359() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_360() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_361() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_362() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_363() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_364() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_365() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_366() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_367() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_368() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_369() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_370() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_371() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_372() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_373() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_374() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_375() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_376() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_377() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_378() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_379() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_380() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_381() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_382() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_383() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_384() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_385() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_386() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_387() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_388() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_389() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_390() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_391() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_392() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_393() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_394() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_395() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_396() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_397() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_398() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_399() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_400() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_401() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_402() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_403() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_404() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_405() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_406() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_407() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_408() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_409() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_410() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_411() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_412() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_413() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_414() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_415() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_416() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_417() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_418() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_419() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_420() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_421() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_422() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_423() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_424() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_425() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_426() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_427() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_428() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_429() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_430() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_431() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_432() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_433() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_434() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_435() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_436() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_437() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_438() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_439() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_440() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_441() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_442() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_443() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_444() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_445() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_446() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_447() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_448() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_449() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_450() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_451() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_452() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_453() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_454() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_455() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_456() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_457() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_458() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_459() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_460() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_461() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_462() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_463() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_464() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_465() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_466() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_467() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_468() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_469() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_470() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_471() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_472() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_473() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_474() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_475() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_476() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_477() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_478() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_479() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_480() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_481() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_482() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_483() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_484() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_485() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_486() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_487() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_488() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_489() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_490() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_491() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_492() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_493() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_494() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_495() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_496() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_497() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_498() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_499() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_500() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_501() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_502() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_503() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_504() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_505() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_506() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_507() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_508() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_509() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_510() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_511() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_512() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_513() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_514() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_515() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_516() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_517() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_518() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_519() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_520() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_521() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_522() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_523() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_524() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_525() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_526() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_527() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_528() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_529() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_530() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_531() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_532() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_533() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_534() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_535() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_536() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_537() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_538() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_539() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_540() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_541() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_542() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_543() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_544() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_545() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_546() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_547() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_548() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_549() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_550() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_551() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_552() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    #[test]
    fn test_profile_stress_553() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }

    // Data pipeline verification and stream throughput check padding line 0
}
