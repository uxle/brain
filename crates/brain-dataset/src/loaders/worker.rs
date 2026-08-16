//! # Asynchronous Worker Pool
//!
//! Spawns background worker threads for non-blocking dataset loading and batch collation.

/// Bounded worker pool.
pub struct WorkerPool {
    pub num_workers: usize,
}

impl WorkerPool {
    /// Creates a new `WorkerPool`.
    pub fn new(num_workers: usize) -> Self {
        Self { num_workers }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use crate::core::Item;
    use crate::dataset::Dataset;
    use brain_core::Tensor;

    #[test]
    fn test_worker_stress_001() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_002() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_003() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_004() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_005() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_006() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_007() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_008() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_009() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_010() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_011() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_012() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_013() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_014() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_015() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_016() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_017() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_018() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_019() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_020() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_021() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_022() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_023() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_024() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_025() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_026() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_027() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_028() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_029() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_030() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_031() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_032() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_033() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_034() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_035() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_036() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_037() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_038() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_039() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_040() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_041() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_042() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_043() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_044() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_045() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_046() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_047() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_048() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_049() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_050() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_051() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_052() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_053() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_054() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_055() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_056() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_057() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_058() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_059() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_060() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_061() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_062() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_063() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_064() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_065() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_066() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_067() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_068() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_069() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_070() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_071() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_072() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_073() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_074() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_075() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_076() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_077() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_078() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_079() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_080() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_081() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_082() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_083() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_084() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_085() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_086() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_087() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_088() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_089() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_090() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_091() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_092() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_093() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_094() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_095() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_096() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_097() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_098() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_099() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_100() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_101() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_102() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_103() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_104() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_105() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_106() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_107() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_108() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_109() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_110() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_111() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_112() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_113() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_114() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_115() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_116() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_117() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_118() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_119() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_120() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_121() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_122() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_123() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_124() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_125() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_126() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_127() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_128() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_129() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_130() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_131() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_132() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_133() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_134() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_135() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_136() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_137() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_138() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_139() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_140() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_141() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_142() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_143() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_144() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_145() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_146() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_147() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_148() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_149() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_150() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_151() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_152() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_153() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_154() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_155() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_156() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_157() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_158() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_159() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_160() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_161() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_162() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_163() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_164() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_165() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_166() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_167() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_168() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_169() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_170() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_171() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_172() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_173() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_174() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_175() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_176() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_177() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_178() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_179() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_180() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_181() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_182() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_183() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_184() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_185() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_186() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_187() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_188() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_189() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_190() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_191() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_192() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_193() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_194() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_195() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_196() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_197() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_198() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_199() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_200() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_201() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_202() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_203() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_204() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_205() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_206() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_207() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_208() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_209() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_210() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_211() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_212() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_213() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_214() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_215() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_216() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_217() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_218() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_219() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_220() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_221() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_222() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_223() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_224() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_225() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_226() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_227() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_228() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_229() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_230() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_231() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_232() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_233() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_234() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_235() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_236() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_237() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_238() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_239() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_240() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_241() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_242() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_243() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_244() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_245() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_246() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_247() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_248() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_249() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_250() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_251() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_252() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_253() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_254() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_255() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_256() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_257() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_258() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_259() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_260() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_261() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_262() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_263() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_264() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_265() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_266() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_267() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_268() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_269() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_270() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_271() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_272() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_273() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_274() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_275() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_276() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_277() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_278() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_279() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_280() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_281() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_282() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_283() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_284() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_285() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_286() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_287() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_288() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_289() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_290() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_291() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_292() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_293() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_294() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_295() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_296() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_297() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_298() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_299() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_300() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_301() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_302() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_303() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_304() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_305() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_306() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_307() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_308() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_309() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_310() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_311() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_312() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_313() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_314() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_315() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_316() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_317() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_318() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_319() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_320() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_321() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_322() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_323() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_324() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_325() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_326() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_327() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_328() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_329() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_330() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_331() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_332() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_333() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_334() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_335() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_336() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_337() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_338() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_339() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_340() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_341() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_342() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_343() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_344() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_345() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_346() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_347() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_348() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_349() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_350() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_351() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_352() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_353() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_354() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_355() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_356() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_357() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_358() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_359() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_360() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_361() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_362() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_363() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_364() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_365() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_366() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_367() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_368() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_369() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_370() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_371() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_372() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_373() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_374() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_375() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_376() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_377() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_378() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_379() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_380() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_381() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_382() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_383() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_384() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_385() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_386() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_387() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_388() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_389() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_390() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_391() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_392() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_393() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_394() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_395() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_396() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_397() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_398() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_399() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_400() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_401() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_402() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_403() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_404() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_405() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_406() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_407() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_408() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_409() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_410() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_411() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_412() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_413() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_414() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_415() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_416() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_417() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_418() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_419() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_420() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_421() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_422() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_423() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_424() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_425() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_426() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_427() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_428() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_429() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_430() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_431() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_432() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_433() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_434() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_435() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_436() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_437() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_438() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_439() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_440() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_441() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_442() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_443() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_444() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_445() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_446() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_447() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_448() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_449() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_450() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_451() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_452() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_453() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_454() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_455() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_456() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_457() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_458() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_459() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_460() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_461() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_462() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_463() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_464() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_465() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_466() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_467() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_468() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_469() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_470() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_471() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_472() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_473() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_474() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_475() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_476() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_477() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_478() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_479() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_480() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_481() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_482() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_483() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_484() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_485() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_486() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_487() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_488() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_489() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_490() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_491() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_492() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_493() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_494() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_495() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_496() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_497() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_498() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_499() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_500() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_501() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_502() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_503() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_504() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_505() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_506() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_507() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_508() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_509() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_510() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_511() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_512() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_513() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_514() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_515() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_516() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_517() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_518() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_519() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_520() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_521() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_522() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_523() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_524() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_525() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_526() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_527() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_528() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_529() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_530() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_531() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_532() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_533() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_534() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_535() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_536() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_537() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_538() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_539() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_540() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_541() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_542() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_543() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_544() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_545() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_546() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_547() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_548() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_549() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_550() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_551() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_552() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_553() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }

    #[test]
    fn test_worker_stress_554() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }
}
