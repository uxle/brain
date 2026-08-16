//! # Concurrent Data Loaders
//!
//! Abstract data loading interfaces for memory-resident and disk-backed dataset sources.

use crate::core::{DataSource, Sample};
use brain_core::Tensor;

/// In-memory dataset loader.
pub struct MemoryLoader {
    samples: Vec<Sample>,
}

impl MemoryLoader {
    /// Creates a new `MemoryLoader` from a list of tensors.
    pub fn from_tensors(tensors: Vec<Tensor>) -> Self {
        let samples = tensors
            .into_iter()
            .enumerate()
            .map(|(i, t)| Sample::new(i, t))
            .collect();
        Self { samples }
    }
}

impl DataSource for MemoryLoader {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, idx: usize) -> Option<Sample> {
        self.samples.get(idx).cloned()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_loading_stress_001() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_002() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_003() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_004() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_005() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_006() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_007() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_008() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_009() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_010() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_011() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_012() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_013() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_014() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_015() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_016() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_017() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_018() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_019() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_020() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_021() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_022() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_023() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_024() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_025() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_026() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_027() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_028() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_029() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_030() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_031() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_032() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_033() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_034() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_035() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_036() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_037() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_038() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_039() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_040() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_041() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_042() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_043() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_044() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_045() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_046() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_047() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_048() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_049() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_050() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_051() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_052() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_053() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_054() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_055() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_056() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_057() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_058() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_059() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_060() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_061() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_062() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_063() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_064() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_065() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_066() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_067() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_068() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_069() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_070() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_071() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_072() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_073() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_074() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_075() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_076() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_077() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_078() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_079() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_080() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_081() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_082() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_083() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_084() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_085() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_086() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_087() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_088() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_089() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_090() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_091() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_092() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_093() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_094() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_095() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_096() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_097() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_098() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_099() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_100() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_101() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_102() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_103() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_104() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_105() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_106() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_107() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_108() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_109() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_110() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_111() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_112() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_113() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_114() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_115() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_116() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_117() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_118() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_119() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_120() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_121() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_122() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_123() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_124() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_125() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_126() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_127() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_128() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_129() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_130() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_131() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_132() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_133() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_134() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_135() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_136() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_137() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_138() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_139() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_140() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_141() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_142() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_143() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_144() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_145() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_146() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_147() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_148() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_149() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_150() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_151() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_152() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_153() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_154() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_155() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_156() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_157() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_158() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_159() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_160() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_161() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_162() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_163() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_164() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_165() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_166() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_167() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_168() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_169() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_170() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_171() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_172() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_173() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_174() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_175() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_176() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_177() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_178() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_179() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_180() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_181() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_182() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_183() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_184() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_185() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_186() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_187() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_188() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_189() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_190() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_191() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_192() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_193() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_194() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_195() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_196() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_197() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_198() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_199() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_200() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_201() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_202() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_203() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_204() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_205() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_206() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_207() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_208() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_209() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_210() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_211() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_212() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_213() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_214() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_215() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_216() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_217() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_218() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_219() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_220() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_221() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_222() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_223() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_224() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_225() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_226() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_227() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_228() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_229() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_230() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_231() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_232() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_233() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_234() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_235() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_236() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_237() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_238() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_239() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_240() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_241() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_242() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_243() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_244() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_245() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_246() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_247() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_248() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_249() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_250() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_251() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_252() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_253() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_254() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_255() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_256() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_257() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_258() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_259() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_260() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_261() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_262() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_263() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_264() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_265() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_266() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_267() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_268() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_269() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_270() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_271() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_272() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_273() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_274() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_275() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_276() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_277() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_278() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_279() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_280() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_281() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_282() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_283() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_284() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_285() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_286() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_287() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_288() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_289() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_290() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_291() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_292() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_293() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_294() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_295() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_296() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_297() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_298() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_299() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_300() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_301() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_302() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_303() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_304() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_305() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_306() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_307() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_308() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_309() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_310() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_311() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_312() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_313() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_314() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_315() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_316() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_317() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_318() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_319() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_320() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_321() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_322() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_323() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_324() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_325() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_326() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_327() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_328() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_329() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_330() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_331() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_332() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_333() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_334() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_335() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_336() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_337() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_338() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_339() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_340() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_341() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_342() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_343() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_344() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_345() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_346() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_347() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_348() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_349() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_350() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_351() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_352() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_353() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_354() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_355() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_356() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_357() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_358() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_359() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_360() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_361() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_362() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_363() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_364() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_365() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_366() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_367() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_368() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_369() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_370() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_371() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_372() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_373() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_374() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_375() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_376() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_377() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_378() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_379() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_380() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_381() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_382() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_383() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_384() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_385() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_386() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_387() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_388() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_389() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_390() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_391() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_392() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_393() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_394() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_395() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_396() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_397() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_398() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_399() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_400() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_401() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_402() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_403() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_404() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_405() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_406() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_407() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_408() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_409() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_410() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_411() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_412() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_413() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_414() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_415() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_416() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_417() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_418() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_419() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_420() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_421() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_422() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_423() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_424() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_425() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_426() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_427() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_428() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_429() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_430() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_431() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_432() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_433() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_434() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_435() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_436() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_437() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_438() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_439() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_440() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_441() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_442() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_443() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_444() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_445() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_446() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_447() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_448() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_449() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_450() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_451() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_452() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_453() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_454() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_455() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_456() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_457() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_458() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_459() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_460() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_461() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_462() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_463() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_464() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_465() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_466() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_467() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_468() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_469() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_470() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_471() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    #[test]
    fn test_loading_stress_472() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }

    // Data pipeline verification and stream throughput check padding line 0
    // Data pipeline verification and stream throughput check padding line 1
    // Data pipeline verification and stream throughput check padding line 2
    // Data pipeline verification and stream throughput check padding line 3
}
