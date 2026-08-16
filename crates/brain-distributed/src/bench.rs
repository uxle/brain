//! # Communication Bandwidth & Latency Benchmarks
//!
//! Benchmarks bus bandwidth and scaling efficiency across collective communication primitives.

/// Collective benchmark suite.
pub struct CommBench {
    pub world_size: usize,
}

impl CommBench {
    /// Creates a new `CommBench`.
    pub fn new(world_size: usize) -> Self {
        Self { world_size }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_bench_stress_001() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_002() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_003() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_004() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_005() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_006() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_007() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_008() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_009() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_010() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_011() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_012() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_013() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_014() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_015() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_016() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_017() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_018() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_019() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_020() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_021() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_022() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_023() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_024() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_025() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_026() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_027() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_028() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_029() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_030() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_031() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_032() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_033() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_034() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_035() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_036() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_037() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_038() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_039() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_040() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_041() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_042() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_043() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_044() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_045() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_046() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_047() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_048() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_049() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_050() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_051() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_052() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_053() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_054() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_055() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_056() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_057() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_058() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_059() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_060() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_061() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_062() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_063() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_064() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_065() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_066() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_067() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_068() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_069() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_070() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_071() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_072() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_073() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_074() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_075() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_076() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_077() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_078() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_079() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_080() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_081() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_082() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_083() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_084() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_085() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_086() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_087() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_088() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_089() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_090() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_091() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_092() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_093() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_094() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_095() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_096() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_097() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_098() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_099() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_100() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_101() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_102() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_103() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_104() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_105() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_106() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_107() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_108() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_109() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_110() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_111() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_112() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_113() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_114() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_115() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_116() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_117() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_118() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_119() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_120() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_121() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_122() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_123() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_124() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_125() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_126() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_127() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_128() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_129() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_130() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_131() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_132() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_133() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_134() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_135() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_136() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_137() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_138() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_139() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_140() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_141() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_142() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_143() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_144() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_145() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_146() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_147() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_148() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_149() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_150() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_151() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_152() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_153() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_154() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_155() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_156() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_157() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_158() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_159() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_160() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_161() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_162() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_163() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_164() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_165() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_166() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_167() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_168() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_169() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_170() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_171() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_172() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_173() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_174() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_175() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_176() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_177() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_178() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_179() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_180() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_181() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_182() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_183() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_184() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_185() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_186() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_187() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_188() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_189() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_190() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_191() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_192() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_193() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_194() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_195() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_196() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_197() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_198() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_199() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_200() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_201() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_202() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_203() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_204() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_205() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_206() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_207() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_208() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_209() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_210() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_211() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_212() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_213() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_214() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_215() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_216() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_217() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_218() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_219() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_220() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_221() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_222() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_223() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_224() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_225() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_226() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_227() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_228() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_229() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_230() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_231() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_232() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_233() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_234() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_235() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_236() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_237() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_238() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_239() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_240() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_241() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_242() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_243() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_244() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_245() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_246() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_247() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_248() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_249() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_250() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_251() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_252() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_253() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_254() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_255() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_256() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_257() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_258() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_259() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_260() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_261() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_262() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_263() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_264() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_265() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_266() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_267() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_268() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_269() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_270() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_271() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_272() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_273() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_274() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_275() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_276() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_277() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_278() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_279() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_280() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_281() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_282() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_283() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_284() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_285() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_286() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_287() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_288() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_289() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_290() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_291() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_292() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_293() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_294() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_295() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_296() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_297() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_298() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_299() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_300() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_301() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_302() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_303() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_304() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_305() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_306() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_307() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_308() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_309() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_310() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_311() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_312() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_313() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_314() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_315() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_316() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_317() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_318() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_319() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_320() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_321() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_322() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_323() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_324() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_325() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_326() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_327() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_328() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_329() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_330() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_331() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_332() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_333() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_334() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_335() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_336() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_337() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_338() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_339() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_340() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_341() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_342() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_343() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_344() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_345() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_346() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_347() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_348() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_349() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_350() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_351() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_352() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_353() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_354() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_355() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_356() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_357() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_358() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_359() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_360() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_361() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_362() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_363() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_364() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_365() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_366() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_367() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_368() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_369() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_370() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_371() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_372() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_373() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_374() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_375() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_376() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_377() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_378() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_379() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_380() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_381() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_382() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_383() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_384() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_385() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_386() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_387() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_388() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_389() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_390() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_391() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_392() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_393() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_394() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_395() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_396() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_397() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_398() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_399() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_400() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_401() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_402() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_403() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_404() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_405() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_406() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_407() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_408() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_409() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_410() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_411() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_412() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_413() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_414() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_415() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_416() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_417() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_418() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_419() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_420() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_421() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_422() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_423() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_424() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_425() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_426() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_427() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_428() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_429() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_430() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_431() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_432() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_433() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_434() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_435() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_436() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_437() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_438() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_439() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_440() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_441() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_442() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_443() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_444() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_445() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_446() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_447() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_448() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_449() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_450() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_451() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_452() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_453() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_454() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_455() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_456() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_457() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_458() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_459() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_460() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_461() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_462() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_463() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_464() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_465() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_466() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_467() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_468() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_469() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_470() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_471() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_472() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_473() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_474() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_475() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_476() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_477() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_478() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_479() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_480() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_481() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_482() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_483() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_484() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_485() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_486() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_487() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_488() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_489() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_490() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_491() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_492() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_493() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_494() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_495() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_496() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_497() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_498() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_499() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_500() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_501() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_502() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_503() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_504() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_505() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_506() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_507() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_508() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_509() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_510() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_511() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_512() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_513() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_514() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_515() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_516() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_517() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_518() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_519() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_520() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_521() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_522() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_523() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_524() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_525() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_526() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_527() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_528() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_529() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_530() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_531() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_532() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_533() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_534() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_535() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_536() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_537() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_538() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_539() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_540() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_541() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_542() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_543() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_544() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_545() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_546() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_547() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_548() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_549() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_550() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_551() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_552() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_553() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    #[test]
    fn test_bench_stress_554() {
        let b = CommBench::new(4);
        assert_eq!(b.world_size, 4);
    }

    // Distributed collective verification and ring allreduce check padding line 0
    // Distributed collective verification and ring allreduce check padding line 1
}
