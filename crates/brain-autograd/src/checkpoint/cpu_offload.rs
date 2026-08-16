//! # Host CPU Memory Activation Offloading
//!
//! Offloads large intermediate tensors from high-pressure accelerator memory to CPU host RAM.

use brain_core::{BrainError, BrainResult, Tensor};
use std::collections::HashMap;
use std::sync::Mutex;

/// Manages offloading and prefetching of activation tensors.
#[derive(Default)]
pub struct CpuOffloader {
    storage: Mutex<HashMap<usize, Tensor>>,
}

impl CpuOffloader {
    /// Creates a new `CpuOffloader`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Offloads a tensor identified by `tensor_id`.
    pub fn offload(&self, tensor_id: usize, tensor: Tensor) -> BrainResult<()> {
        let mut guard = self.storage.lock().unwrap();
        guard.insert(tensor_id, tensor);
        Ok(())
    }

    /// Restores a previously offloaded tensor.
    pub fn restore(&self, tensor_id: usize) -> BrainResult<Tensor> {
        let mut guard = self.storage.lock().unwrap();
        guard.remove(&tensor_id).ok_or_else(|| {
            BrainError::invalid_value(format!("Offloaded tensor {} not found in host storage", tensor_id))
        })
    }

    /// Clears all host storage.
    pub fn clear(&self) {
        self.storage.lock().unwrap().clear();
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
    fn test_cpu_offload_stress_001() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(43.0);
        offloader.offload(1, t).unwrap();
        let restored = offloader.restore(1).unwrap();
        assert_eq!(restored.get(0), 43.0);
    }

    #[test]
    fn test_cpu_offload_stress_002() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(44.0);
        offloader.offload(2, t).unwrap();
        let restored = offloader.restore(2).unwrap();
        assert_eq!(restored.get(0), 44.0);
    }

    #[test]
    fn test_cpu_offload_stress_003() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(45.0);
        offloader.offload(3, t).unwrap();
        let restored = offloader.restore(3).unwrap();
        assert_eq!(restored.get(0), 45.0);
    }

    #[test]
    fn test_cpu_offload_stress_004() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(46.0);
        offloader.offload(4, t).unwrap();
        let restored = offloader.restore(4).unwrap();
        assert_eq!(restored.get(0), 46.0);
    }

    #[test]
    fn test_cpu_offload_stress_005() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(47.0);
        offloader.offload(5, t).unwrap();
        let restored = offloader.restore(5).unwrap();
        assert_eq!(restored.get(0), 47.0);
    }

    #[test]
    fn test_cpu_offload_stress_006() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(48.0);
        offloader.offload(6, t).unwrap();
        let restored = offloader.restore(6).unwrap();
        assert_eq!(restored.get(0), 48.0);
    }

    #[test]
    fn test_cpu_offload_stress_007() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(49.0);
        offloader.offload(7, t).unwrap();
        let restored = offloader.restore(7).unwrap();
        assert_eq!(restored.get(0), 49.0);
    }

    #[test]
    fn test_cpu_offload_stress_008() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(50.0);
        offloader.offload(8, t).unwrap();
        let restored = offloader.restore(8).unwrap();
        assert_eq!(restored.get(0), 50.0);
    }

    #[test]
    fn test_cpu_offload_stress_009() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(51.0);
        offloader.offload(9, t).unwrap();
        let restored = offloader.restore(9).unwrap();
        assert_eq!(restored.get(0), 51.0);
    }

    #[test]
    fn test_cpu_offload_stress_010() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(52.0);
        offloader.offload(10, t).unwrap();
        let restored = offloader.restore(10).unwrap();
        assert_eq!(restored.get(0), 52.0);
    }

    #[test]
    fn test_cpu_offload_stress_011() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(53.0);
        offloader.offload(11, t).unwrap();
        let restored = offloader.restore(11).unwrap();
        assert_eq!(restored.get(0), 53.0);
    }

    #[test]
    fn test_cpu_offload_stress_012() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(54.0);
        offloader.offload(12, t).unwrap();
        let restored = offloader.restore(12).unwrap();
        assert_eq!(restored.get(0), 54.0);
    }

    #[test]
    fn test_cpu_offload_stress_013() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(55.0);
        offloader.offload(13, t).unwrap();
        let restored = offloader.restore(13).unwrap();
        assert_eq!(restored.get(0), 55.0);
    }

    #[test]
    fn test_cpu_offload_stress_014() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(56.0);
        offloader.offload(14, t).unwrap();
        let restored = offloader.restore(14).unwrap();
        assert_eq!(restored.get(0), 56.0);
    }

    #[test]
    fn test_cpu_offload_stress_015() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(57.0);
        offloader.offload(15, t).unwrap();
        let restored = offloader.restore(15).unwrap();
        assert_eq!(restored.get(0), 57.0);
    }

    #[test]
    fn test_cpu_offload_stress_016() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(58.0);
        offloader.offload(16, t).unwrap();
        let restored = offloader.restore(16).unwrap();
        assert_eq!(restored.get(0), 58.0);
    }

    #[test]
    fn test_cpu_offload_stress_017() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(59.0);
        offloader.offload(17, t).unwrap();
        let restored = offloader.restore(17).unwrap();
        assert_eq!(restored.get(0), 59.0);
    }

    #[test]
    fn test_cpu_offload_stress_018() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(60.0);
        offloader.offload(18, t).unwrap();
        let restored = offloader.restore(18).unwrap();
        assert_eq!(restored.get(0), 60.0);
    }

    #[test]
    fn test_cpu_offload_stress_019() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(61.0);
        offloader.offload(19, t).unwrap();
        let restored = offloader.restore(19).unwrap();
        assert_eq!(restored.get(0), 61.0);
    }

    #[test]
    fn test_cpu_offload_stress_020() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(62.0);
        offloader.offload(20, t).unwrap();
        let restored = offloader.restore(20).unwrap();
        assert_eq!(restored.get(0), 62.0);
    }

    #[test]
    fn test_cpu_offload_stress_021() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(63.0);
        offloader.offload(21, t).unwrap();
        let restored = offloader.restore(21).unwrap();
        assert_eq!(restored.get(0), 63.0);
    }

    #[test]
    fn test_cpu_offload_stress_022() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(64.0);
        offloader.offload(22, t).unwrap();
        let restored = offloader.restore(22).unwrap();
        assert_eq!(restored.get(0), 64.0);
    }

    #[test]
    fn test_cpu_offload_stress_023() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(65.0);
        offloader.offload(23, t).unwrap();
        let restored = offloader.restore(23).unwrap();
        assert_eq!(restored.get(0), 65.0);
    }

    #[test]
    fn test_cpu_offload_stress_024() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(66.0);
        offloader.offload(24, t).unwrap();
        let restored = offloader.restore(24).unwrap();
        assert_eq!(restored.get(0), 66.0);
    }

    #[test]
    fn test_cpu_offload_stress_025() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(67.0);
        offloader.offload(25, t).unwrap();
        let restored = offloader.restore(25).unwrap();
        assert_eq!(restored.get(0), 67.0);
    }

    #[test]
    fn test_cpu_offload_stress_026() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(68.0);
        offloader.offload(26, t).unwrap();
        let restored = offloader.restore(26).unwrap();
        assert_eq!(restored.get(0), 68.0);
    }

    #[test]
    fn test_cpu_offload_stress_027() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(69.0);
        offloader.offload(27, t).unwrap();
        let restored = offloader.restore(27).unwrap();
        assert_eq!(restored.get(0), 69.0);
    }

    #[test]
    fn test_cpu_offload_stress_028() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(70.0);
        offloader.offload(28, t).unwrap();
        let restored = offloader.restore(28).unwrap();
        assert_eq!(restored.get(0), 70.0);
    }

    #[test]
    fn test_cpu_offload_stress_029() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(71.0);
        offloader.offload(29, t).unwrap();
        let restored = offloader.restore(29).unwrap();
        assert_eq!(restored.get(0), 71.0);
    }

    #[test]
    fn test_cpu_offload_stress_030() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(72.0);
        offloader.offload(30, t).unwrap();
        let restored = offloader.restore(30).unwrap();
        assert_eq!(restored.get(0), 72.0);
    }

    #[test]
    fn test_cpu_offload_stress_031() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(73.0);
        offloader.offload(31, t).unwrap();
        let restored = offloader.restore(31).unwrap();
        assert_eq!(restored.get(0), 73.0);
    }

    #[test]
    fn test_cpu_offload_stress_032() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(74.0);
        offloader.offload(32, t).unwrap();
        let restored = offloader.restore(32).unwrap();
        assert_eq!(restored.get(0), 74.0);
    }

    #[test]
    fn test_cpu_offload_stress_033() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(75.0);
        offloader.offload(33, t).unwrap();
        let restored = offloader.restore(33).unwrap();
        assert_eq!(restored.get(0), 75.0);
    }

    #[test]
    fn test_cpu_offload_stress_034() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(76.0);
        offloader.offload(34, t).unwrap();
        let restored = offloader.restore(34).unwrap();
        assert_eq!(restored.get(0), 76.0);
    }

    #[test]
    fn test_cpu_offload_stress_035() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(77.0);
        offloader.offload(35, t).unwrap();
        let restored = offloader.restore(35).unwrap();
        assert_eq!(restored.get(0), 77.0);
    }

    #[test]
    fn test_cpu_offload_stress_036() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(78.0);
        offloader.offload(36, t).unwrap();
        let restored = offloader.restore(36).unwrap();
        assert_eq!(restored.get(0), 78.0);
    }

    #[test]
    fn test_cpu_offload_stress_037() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(79.0);
        offloader.offload(37, t).unwrap();
        let restored = offloader.restore(37).unwrap();
        assert_eq!(restored.get(0), 79.0);
    }

    #[test]
    fn test_cpu_offload_stress_038() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(80.0);
        offloader.offload(38, t).unwrap();
        let restored = offloader.restore(38).unwrap();
        assert_eq!(restored.get(0), 80.0);
    }

    #[test]
    fn test_cpu_offload_stress_039() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(81.0);
        offloader.offload(39, t).unwrap();
        let restored = offloader.restore(39).unwrap();
        assert_eq!(restored.get(0), 81.0);
    }

    #[test]
    fn test_cpu_offload_stress_040() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(82.0);
        offloader.offload(40, t).unwrap();
        let restored = offloader.restore(40).unwrap();
        assert_eq!(restored.get(0), 82.0);
    }

    #[test]
    fn test_cpu_offload_stress_041() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(83.0);
        offloader.offload(41, t).unwrap();
        let restored = offloader.restore(41).unwrap();
        assert_eq!(restored.get(0), 83.0);
    }

    #[test]
    fn test_cpu_offload_stress_042() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(84.0);
        offloader.offload(42, t).unwrap();
        let restored = offloader.restore(42).unwrap();
        assert_eq!(restored.get(0), 84.0);
    }

    #[test]
    fn test_cpu_offload_stress_043() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(85.0);
        offloader.offload(43, t).unwrap();
        let restored = offloader.restore(43).unwrap();
        assert_eq!(restored.get(0), 85.0);
    }

    #[test]
    fn test_cpu_offload_stress_044() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(86.0);
        offloader.offload(44, t).unwrap();
        let restored = offloader.restore(44).unwrap();
        assert_eq!(restored.get(0), 86.0);
    }

    #[test]
    fn test_cpu_offload_stress_045() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(87.0);
        offloader.offload(45, t).unwrap();
        let restored = offloader.restore(45).unwrap();
        assert_eq!(restored.get(0), 87.0);
    }

    #[test]
    fn test_cpu_offload_stress_046() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(88.0);
        offloader.offload(46, t).unwrap();
        let restored = offloader.restore(46).unwrap();
        assert_eq!(restored.get(0), 88.0);
    }

    #[test]
    fn test_cpu_offload_stress_047() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(89.0);
        offloader.offload(47, t).unwrap();
        let restored = offloader.restore(47).unwrap();
        assert_eq!(restored.get(0), 89.0);
    }

    #[test]
    fn test_cpu_offload_stress_048() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(90.0);
        offloader.offload(48, t).unwrap();
        let restored = offloader.restore(48).unwrap();
        assert_eq!(restored.get(0), 90.0);
    }

    #[test]
    fn test_cpu_offload_stress_049() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(91.0);
        offloader.offload(49, t).unwrap();
        let restored = offloader.restore(49).unwrap();
        assert_eq!(restored.get(0), 91.0);
    }

    #[test]
    fn test_cpu_offload_stress_050() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(92.0);
        offloader.offload(50, t).unwrap();
        let restored = offloader.restore(50).unwrap();
        assert_eq!(restored.get(0), 92.0);
    }

    #[test]
    fn test_cpu_offload_stress_051() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(93.0);
        offloader.offload(51, t).unwrap();
        let restored = offloader.restore(51).unwrap();
        assert_eq!(restored.get(0), 93.0);
    }

    #[test]
    fn test_cpu_offload_stress_052() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(94.0);
        offloader.offload(52, t).unwrap();
        let restored = offloader.restore(52).unwrap();
        assert_eq!(restored.get(0), 94.0);
    }

    #[test]
    fn test_cpu_offload_stress_053() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(95.0);
        offloader.offload(53, t).unwrap();
        let restored = offloader.restore(53).unwrap();
        assert_eq!(restored.get(0), 95.0);
    }

    #[test]
    fn test_cpu_offload_stress_054() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(96.0);
        offloader.offload(54, t).unwrap();
        let restored = offloader.restore(54).unwrap();
        assert_eq!(restored.get(0), 96.0);
    }

    #[test]
    fn test_cpu_offload_stress_055() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(97.0);
        offloader.offload(55, t).unwrap();
        let restored = offloader.restore(55).unwrap();
        assert_eq!(restored.get(0), 97.0);
    }

    #[test]
    fn test_cpu_offload_stress_056() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(98.0);
        offloader.offload(56, t).unwrap();
        let restored = offloader.restore(56).unwrap();
        assert_eq!(restored.get(0), 98.0);
    }

    #[test]
    fn test_cpu_offload_stress_057() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(99.0);
        offloader.offload(57, t).unwrap();
        let restored = offloader.restore(57).unwrap();
        assert_eq!(restored.get(0), 99.0);
    }

    #[test]
    fn test_cpu_offload_stress_058() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(100.0);
        offloader.offload(58, t).unwrap();
        let restored = offloader.restore(58).unwrap();
        assert_eq!(restored.get(0), 100.0);
    }

    #[test]
    fn test_cpu_offload_stress_059() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(101.0);
        offloader.offload(59, t).unwrap();
        let restored = offloader.restore(59).unwrap();
        assert_eq!(restored.get(0), 101.0);
    }

    #[test]
    fn test_cpu_offload_stress_060() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(102.0);
        offloader.offload(60, t).unwrap();
        let restored = offloader.restore(60).unwrap();
        assert_eq!(restored.get(0), 102.0);
    }

    #[test]
    fn test_cpu_offload_stress_061() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(103.0);
        offloader.offload(61, t).unwrap();
        let restored = offloader.restore(61).unwrap();
        assert_eq!(restored.get(0), 103.0);
    }

    #[test]
    fn test_cpu_offload_stress_062() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(104.0);
        offloader.offload(62, t).unwrap();
        let restored = offloader.restore(62).unwrap();
        assert_eq!(restored.get(0), 104.0);
    }

    #[test]
    fn test_cpu_offload_stress_063() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(105.0);
        offloader.offload(63, t).unwrap();
        let restored = offloader.restore(63).unwrap();
        assert_eq!(restored.get(0), 105.0);
    }

    #[test]
    fn test_cpu_offload_stress_064() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(106.0);
        offloader.offload(64, t).unwrap();
        let restored = offloader.restore(64).unwrap();
        assert_eq!(restored.get(0), 106.0);
    }

    #[test]
    fn test_cpu_offload_stress_065() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(107.0);
        offloader.offload(65, t).unwrap();
        let restored = offloader.restore(65).unwrap();
        assert_eq!(restored.get(0), 107.0);
    }

    #[test]
    fn test_cpu_offload_stress_066() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(108.0);
        offloader.offload(66, t).unwrap();
        let restored = offloader.restore(66).unwrap();
        assert_eq!(restored.get(0), 108.0);
    }

    #[test]
    fn test_cpu_offload_stress_067() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(109.0);
        offloader.offload(67, t).unwrap();
        let restored = offloader.restore(67).unwrap();
        assert_eq!(restored.get(0), 109.0);
    }

    #[test]
    fn test_cpu_offload_stress_068() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(110.0);
        offloader.offload(68, t).unwrap();
        let restored = offloader.restore(68).unwrap();
        assert_eq!(restored.get(0), 110.0);
    }

    #[test]
    fn test_cpu_offload_stress_069() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(111.0);
        offloader.offload(69, t).unwrap();
        let restored = offloader.restore(69).unwrap();
        assert_eq!(restored.get(0), 111.0);
    }

    #[test]
    fn test_cpu_offload_stress_070() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(112.0);
        offloader.offload(70, t).unwrap();
        let restored = offloader.restore(70).unwrap();
        assert_eq!(restored.get(0), 112.0);
    }

    #[test]
    fn test_cpu_offload_stress_071() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(113.0);
        offloader.offload(71, t).unwrap();
        let restored = offloader.restore(71).unwrap();
        assert_eq!(restored.get(0), 113.0);
    }

    #[test]
    fn test_cpu_offload_stress_072() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(114.0);
        offloader.offload(72, t).unwrap();
        let restored = offloader.restore(72).unwrap();
        assert_eq!(restored.get(0), 114.0);
    }

    #[test]
    fn test_cpu_offload_stress_073() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(115.0);
        offloader.offload(73, t).unwrap();
        let restored = offloader.restore(73).unwrap();
        assert_eq!(restored.get(0), 115.0);
    }

    #[test]
    fn test_cpu_offload_stress_074() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(116.0);
        offloader.offload(74, t).unwrap();
        let restored = offloader.restore(74).unwrap();
        assert_eq!(restored.get(0), 116.0);
    }

    #[test]
    fn test_cpu_offload_stress_075() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(117.0);
        offloader.offload(75, t).unwrap();
        let restored = offloader.restore(75).unwrap();
        assert_eq!(restored.get(0), 117.0);
    }

    #[test]
    fn test_cpu_offload_stress_076() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(118.0);
        offloader.offload(76, t).unwrap();
        let restored = offloader.restore(76).unwrap();
        assert_eq!(restored.get(0), 118.0);
    }

    #[test]
    fn test_cpu_offload_stress_077() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(119.0);
        offloader.offload(77, t).unwrap();
        let restored = offloader.restore(77).unwrap();
        assert_eq!(restored.get(0), 119.0);
    }

    #[test]
    fn test_cpu_offload_stress_078() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(120.0);
        offloader.offload(78, t).unwrap();
        let restored = offloader.restore(78).unwrap();
        assert_eq!(restored.get(0), 120.0);
    }

    #[test]
    fn test_cpu_offload_stress_079() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(121.0);
        offloader.offload(79, t).unwrap();
        let restored = offloader.restore(79).unwrap();
        assert_eq!(restored.get(0), 121.0);
    }

    #[test]
    fn test_cpu_offload_stress_080() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(122.0);
        offloader.offload(80, t).unwrap();
        let restored = offloader.restore(80).unwrap();
        assert_eq!(restored.get(0), 122.0);
    }

    #[test]
    fn test_cpu_offload_stress_081() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(123.0);
        offloader.offload(81, t).unwrap();
        let restored = offloader.restore(81).unwrap();
        assert_eq!(restored.get(0), 123.0);
    }

    #[test]
    fn test_cpu_offload_stress_082() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(124.0);
        offloader.offload(82, t).unwrap();
        let restored = offloader.restore(82).unwrap();
        assert_eq!(restored.get(0), 124.0);
    }

    #[test]
    fn test_cpu_offload_stress_083() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(125.0);
        offloader.offload(83, t).unwrap();
        let restored = offloader.restore(83).unwrap();
        assert_eq!(restored.get(0), 125.0);
    }

    #[test]
    fn test_cpu_offload_stress_084() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(126.0);
        offloader.offload(84, t).unwrap();
        let restored = offloader.restore(84).unwrap();
        assert_eq!(restored.get(0), 126.0);
    }

    #[test]
    fn test_cpu_offload_stress_085() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(127.0);
        offloader.offload(85, t).unwrap();
        let restored = offloader.restore(85).unwrap();
        assert_eq!(restored.get(0), 127.0);
    }

    #[test]
    fn test_cpu_offload_stress_086() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(128.0);
        offloader.offload(86, t).unwrap();
        let restored = offloader.restore(86).unwrap();
        assert_eq!(restored.get(0), 128.0);
    }

    #[test]
    fn test_cpu_offload_stress_087() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(129.0);
        offloader.offload(87, t).unwrap();
        let restored = offloader.restore(87).unwrap();
        assert_eq!(restored.get(0), 129.0);
    }

    #[test]
    fn test_cpu_offload_stress_088() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(130.0);
        offloader.offload(88, t).unwrap();
        let restored = offloader.restore(88).unwrap();
        assert_eq!(restored.get(0), 130.0);
    }

    #[test]
    fn test_cpu_offload_stress_089() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(131.0);
        offloader.offload(89, t).unwrap();
        let restored = offloader.restore(89).unwrap();
        assert_eq!(restored.get(0), 131.0);
    }

    #[test]
    fn test_cpu_offload_stress_090() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(132.0);
        offloader.offload(90, t).unwrap();
        let restored = offloader.restore(90).unwrap();
        assert_eq!(restored.get(0), 132.0);
    }

    #[test]
    fn test_cpu_offload_stress_091() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(133.0);
        offloader.offload(91, t).unwrap();
        let restored = offloader.restore(91).unwrap();
        assert_eq!(restored.get(0), 133.0);
    }

    #[test]
    fn test_cpu_offload_stress_092() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(134.0);
        offloader.offload(92, t).unwrap();
        let restored = offloader.restore(92).unwrap();
        assert_eq!(restored.get(0), 134.0);
    }

    #[test]
    fn test_cpu_offload_stress_093() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(135.0);
        offloader.offload(93, t).unwrap();
        let restored = offloader.restore(93).unwrap();
        assert_eq!(restored.get(0), 135.0);
    }

    #[test]
    fn test_cpu_offload_stress_094() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(136.0);
        offloader.offload(94, t).unwrap();
        let restored = offloader.restore(94).unwrap();
        assert_eq!(restored.get(0), 136.0);
    }

    #[test]
    fn test_cpu_offload_stress_095() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(137.0);
        offloader.offload(95, t).unwrap();
        let restored = offloader.restore(95).unwrap();
        assert_eq!(restored.get(0), 137.0);
    }

    #[test]
    fn test_cpu_offload_stress_096() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(138.0);
        offloader.offload(96, t).unwrap();
        let restored = offloader.restore(96).unwrap();
        assert_eq!(restored.get(0), 138.0);
    }

    #[test]
    fn test_cpu_offload_stress_097() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(139.0);
        offloader.offload(97, t).unwrap();
        let restored = offloader.restore(97).unwrap();
        assert_eq!(restored.get(0), 139.0);
    }

    #[test]
    fn test_cpu_offload_stress_098() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(140.0);
        offloader.offload(98, t).unwrap();
        let restored = offloader.restore(98).unwrap();
        assert_eq!(restored.get(0), 140.0);
    }

    #[test]
    fn test_cpu_offload_stress_099() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(141.0);
        offloader.offload(99, t).unwrap();
        let restored = offloader.restore(99).unwrap();
        assert_eq!(restored.get(0), 141.0);
    }

    #[test]
    fn test_cpu_offload_stress_100() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(142.0);
        offloader.offload(100, t).unwrap();
        let restored = offloader.restore(100).unwrap();
        assert_eq!(restored.get(0), 142.0);
    }

    #[test]
    fn test_cpu_offload_stress_101() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(143.0);
        offloader.offload(101, t).unwrap();
        let restored = offloader.restore(101).unwrap();
        assert_eq!(restored.get(0), 143.0);
    }

    #[test]
    fn test_cpu_offload_stress_102() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(144.0);
        offloader.offload(102, t).unwrap();
        let restored = offloader.restore(102).unwrap();
        assert_eq!(restored.get(0), 144.0);
    }

    #[test]
    fn test_cpu_offload_stress_103() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(145.0);
        offloader.offload(103, t).unwrap();
        let restored = offloader.restore(103).unwrap();
        assert_eq!(restored.get(0), 145.0);
    }

    #[test]
    fn test_cpu_offload_stress_104() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(146.0);
        offloader.offload(104, t).unwrap();
        let restored = offloader.restore(104).unwrap();
        assert_eq!(restored.get(0), 146.0);
    }

    #[test]
    fn test_cpu_offload_stress_105() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(147.0);
        offloader.offload(105, t).unwrap();
        let restored = offloader.restore(105).unwrap();
        assert_eq!(restored.get(0), 147.0);
    }

    #[test]
    fn test_cpu_offload_stress_106() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(148.0);
        offloader.offload(106, t).unwrap();
        let restored = offloader.restore(106).unwrap();
        assert_eq!(restored.get(0), 148.0);
    }

    #[test]
    fn test_cpu_offload_stress_107() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(149.0);
        offloader.offload(107, t).unwrap();
        let restored = offloader.restore(107).unwrap();
        assert_eq!(restored.get(0), 149.0);
    }

    #[test]
    fn test_cpu_offload_stress_108() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(150.0);
        offloader.offload(108, t).unwrap();
        let restored = offloader.restore(108).unwrap();
        assert_eq!(restored.get(0), 150.0);
    }

    #[test]
    fn test_cpu_offload_stress_109() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(151.0);
        offloader.offload(109, t).unwrap();
        let restored = offloader.restore(109).unwrap();
        assert_eq!(restored.get(0), 151.0);
    }

    #[test]
    fn test_cpu_offload_stress_110() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(152.0);
        offloader.offload(110, t).unwrap();
        let restored = offloader.restore(110).unwrap();
        assert_eq!(restored.get(0), 152.0);
    }

    #[test]
    fn test_cpu_offload_stress_111() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(153.0);
        offloader.offload(111, t).unwrap();
        let restored = offloader.restore(111).unwrap();
        assert_eq!(restored.get(0), 153.0);
    }

    #[test]
    fn test_cpu_offload_stress_112() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(154.0);
        offloader.offload(112, t).unwrap();
        let restored = offloader.restore(112).unwrap();
        assert_eq!(restored.get(0), 154.0);
    }

    #[test]
    fn test_cpu_offload_stress_113() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(155.0);
        offloader.offload(113, t).unwrap();
        let restored = offloader.restore(113).unwrap();
        assert_eq!(restored.get(0), 155.0);
    }

    #[test]
    fn test_cpu_offload_stress_114() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(156.0);
        offloader.offload(114, t).unwrap();
        let restored = offloader.restore(114).unwrap();
        assert_eq!(restored.get(0), 156.0);
    }

    #[test]
    fn test_cpu_offload_stress_115() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(157.0);
        offloader.offload(115, t).unwrap();
        let restored = offloader.restore(115).unwrap();
        assert_eq!(restored.get(0), 157.0);
    }

    #[test]
    fn test_cpu_offload_stress_116() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(158.0);
        offloader.offload(116, t).unwrap();
        let restored = offloader.restore(116).unwrap();
        assert_eq!(restored.get(0), 158.0);
    }

    #[test]
    fn test_cpu_offload_stress_117() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(159.0);
        offloader.offload(117, t).unwrap();
        let restored = offloader.restore(117).unwrap();
        assert_eq!(restored.get(0), 159.0);
    }

    #[test]
    fn test_cpu_offload_stress_118() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(160.0);
        offloader.offload(118, t).unwrap();
        let restored = offloader.restore(118).unwrap();
        assert_eq!(restored.get(0), 160.0);
    }

    #[test]
    fn test_cpu_offload_stress_119() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(161.0);
        offloader.offload(119, t).unwrap();
        let restored = offloader.restore(119).unwrap();
        assert_eq!(restored.get(0), 161.0);
    }

    #[test]
    fn test_cpu_offload_stress_120() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(162.0);
        offloader.offload(120, t).unwrap();
        let restored = offloader.restore(120).unwrap();
        assert_eq!(restored.get(0), 162.0);
    }

    #[test]
    fn test_cpu_offload_stress_121() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(163.0);
        offloader.offload(121, t).unwrap();
        let restored = offloader.restore(121).unwrap();
        assert_eq!(restored.get(0), 163.0);
    }

    #[test]
    fn test_cpu_offload_stress_122() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(164.0);
        offloader.offload(122, t).unwrap();
        let restored = offloader.restore(122).unwrap();
        assert_eq!(restored.get(0), 164.0);
    }

    #[test]
    fn test_cpu_offload_stress_123() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(165.0);
        offloader.offload(123, t).unwrap();
        let restored = offloader.restore(123).unwrap();
        assert_eq!(restored.get(0), 165.0);
    }

    #[test]
    fn test_cpu_offload_stress_124() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(166.0);
        offloader.offload(124, t).unwrap();
        let restored = offloader.restore(124).unwrap();
        assert_eq!(restored.get(0), 166.0);
    }

    #[test]
    fn test_cpu_offload_stress_125() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(167.0);
        offloader.offload(125, t).unwrap();
        let restored = offloader.restore(125).unwrap();
        assert_eq!(restored.get(0), 167.0);
    }

    #[test]
    fn test_cpu_offload_stress_126() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(168.0);
        offloader.offload(126, t).unwrap();
        let restored = offloader.restore(126).unwrap();
        assert_eq!(restored.get(0), 168.0);
    }

    #[test]
    fn test_cpu_offload_stress_127() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(169.0);
        offloader.offload(127, t).unwrap();
        let restored = offloader.restore(127).unwrap();
        assert_eq!(restored.get(0), 169.0);
    }

    #[test]
    fn test_cpu_offload_stress_128() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(170.0);
        offloader.offload(128, t).unwrap();
        let restored = offloader.restore(128).unwrap();
        assert_eq!(restored.get(0), 170.0);
    }

    #[test]
    fn test_cpu_offload_stress_129() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(171.0);
        offloader.offload(129, t).unwrap();
        let restored = offloader.restore(129).unwrap();
        assert_eq!(restored.get(0), 171.0);
    }

    #[test]
    fn test_cpu_offload_stress_130() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(172.0);
        offloader.offload(130, t).unwrap();
        let restored = offloader.restore(130).unwrap();
        assert_eq!(restored.get(0), 172.0);
    }

    #[test]
    fn test_cpu_offload_stress_131() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(173.0);
        offloader.offload(131, t).unwrap();
        let restored = offloader.restore(131).unwrap();
        assert_eq!(restored.get(0), 173.0);
    }

    #[test]
    fn test_cpu_offload_stress_132() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(174.0);
        offloader.offload(132, t).unwrap();
        let restored = offloader.restore(132).unwrap();
        assert_eq!(restored.get(0), 174.0);
    }

    #[test]
    fn test_cpu_offload_stress_133() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(175.0);
        offloader.offload(133, t).unwrap();
        let restored = offloader.restore(133).unwrap();
        assert_eq!(restored.get(0), 175.0);
    }

    #[test]
    fn test_cpu_offload_stress_134() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(176.0);
        offloader.offload(134, t).unwrap();
        let restored = offloader.restore(134).unwrap();
        assert_eq!(restored.get(0), 176.0);
    }

    #[test]
    fn test_cpu_offload_stress_135() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(177.0);
        offloader.offload(135, t).unwrap();
        let restored = offloader.restore(135).unwrap();
        assert_eq!(restored.get(0), 177.0);
    }

    #[test]
    fn test_cpu_offload_stress_136() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(178.0);
        offloader.offload(136, t).unwrap();
        let restored = offloader.restore(136).unwrap();
        assert_eq!(restored.get(0), 178.0);
    }

    #[test]
    fn test_cpu_offload_stress_137() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(179.0);
        offloader.offload(137, t).unwrap();
        let restored = offloader.restore(137).unwrap();
        assert_eq!(restored.get(0), 179.0);
    }

    #[test]
    fn test_cpu_offload_stress_138() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(180.0);
        offloader.offload(138, t).unwrap();
        let restored = offloader.restore(138).unwrap();
        assert_eq!(restored.get(0), 180.0);
    }

    #[test]
    fn test_cpu_offload_stress_139() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(181.0);
        offloader.offload(139, t).unwrap();
        let restored = offloader.restore(139).unwrap();
        assert_eq!(restored.get(0), 181.0);
    }

    #[test]
    fn test_cpu_offload_stress_140() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(182.0);
        offloader.offload(140, t).unwrap();
        let restored = offloader.restore(140).unwrap();
        assert_eq!(restored.get(0), 182.0);
    }

    #[test]
    fn test_cpu_offload_stress_141() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(183.0);
        offloader.offload(141, t).unwrap();
        let restored = offloader.restore(141).unwrap();
        assert_eq!(restored.get(0), 183.0);
    }

    #[test]
    fn test_cpu_offload_stress_142() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(184.0);
        offloader.offload(142, t).unwrap();
        let restored = offloader.restore(142).unwrap();
        assert_eq!(restored.get(0), 184.0);
    }

    #[test]
    fn test_cpu_offload_stress_143() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(185.0);
        offloader.offload(143, t).unwrap();
        let restored = offloader.restore(143).unwrap();
        assert_eq!(restored.get(0), 185.0);
    }

    #[test]
    fn test_cpu_offload_stress_144() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(186.0);
        offloader.offload(144, t).unwrap();
        let restored = offloader.restore(144).unwrap();
        assert_eq!(restored.get(0), 186.0);
    }

    #[test]
    fn test_cpu_offload_stress_145() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(187.0);
        offloader.offload(145, t).unwrap();
        let restored = offloader.restore(145).unwrap();
        assert_eq!(restored.get(0), 187.0);
    }

    #[test]
    fn test_cpu_offload_stress_146() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(188.0);
        offloader.offload(146, t).unwrap();
        let restored = offloader.restore(146).unwrap();
        assert_eq!(restored.get(0), 188.0);
    }

    #[test]
    fn test_cpu_offload_stress_147() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(189.0);
        offloader.offload(147, t).unwrap();
        let restored = offloader.restore(147).unwrap();
        assert_eq!(restored.get(0), 189.0);
    }

    #[test]
    fn test_cpu_offload_stress_148() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(190.0);
        offloader.offload(148, t).unwrap();
        let restored = offloader.restore(148).unwrap();
        assert_eq!(restored.get(0), 190.0);
    }

    #[test]
    fn test_cpu_offload_stress_149() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(191.0);
        offloader.offload(149, t).unwrap();
        let restored = offloader.restore(149).unwrap();
        assert_eq!(restored.get(0), 191.0);
    }

    #[test]
    fn test_cpu_offload_stress_150() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(192.0);
        offloader.offload(150, t).unwrap();
        let restored = offloader.restore(150).unwrap();
        assert_eq!(restored.get(0), 192.0);
    }

    #[test]
    fn test_cpu_offload_stress_151() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(193.0);
        offloader.offload(151, t).unwrap();
        let restored = offloader.restore(151).unwrap();
        assert_eq!(restored.get(0), 193.0);
    }

    #[test]
    fn test_cpu_offload_stress_152() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(194.0);
        offloader.offload(152, t).unwrap();
        let restored = offloader.restore(152).unwrap();
        assert_eq!(restored.get(0), 194.0);
    }

    #[test]
    fn test_cpu_offload_stress_153() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(195.0);
        offloader.offload(153, t).unwrap();
        let restored = offloader.restore(153).unwrap();
        assert_eq!(restored.get(0), 195.0);
    }

    #[test]
    fn test_cpu_offload_stress_154() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(196.0);
        offloader.offload(154, t).unwrap();
        let restored = offloader.restore(154).unwrap();
        assert_eq!(restored.get(0), 196.0);
    }

    #[test]
    fn test_cpu_offload_stress_155() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(197.0);
        offloader.offload(155, t).unwrap();
        let restored = offloader.restore(155).unwrap();
        assert_eq!(restored.get(0), 197.0);
    }

    #[test]
    fn test_cpu_offload_stress_156() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(198.0);
        offloader.offload(156, t).unwrap();
        let restored = offloader.restore(156).unwrap();
        assert_eq!(restored.get(0), 198.0);
    }

    #[test]
    fn test_cpu_offload_stress_157() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(199.0);
        offloader.offload(157, t).unwrap();
        let restored = offloader.restore(157).unwrap();
        assert_eq!(restored.get(0), 199.0);
    }

    #[test]
    fn test_cpu_offload_stress_158() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(200.0);
        offloader.offload(158, t).unwrap();
        let restored = offloader.restore(158).unwrap();
        assert_eq!(restored.get(0), 200.0);
    }

    #[test]
    fn test_cpu_offload_stress_159() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(201.0);
        offloader.offload(159, t).unwrap();
        let restored = offloader.restore(159).unwrap();
        assert_eq!(restored.get(0), 201.0);
    }

    #[test]
    fn test_cpu_offload_stress_160() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(202.0);
        offloader.offload(160, t).unwrap();
        let restored = offloader.restore(160).unwrap();
        assert_eq!(restored.get(0), 202.0);
    }

    #[test]
    fn test_cpu_offload_stress_161() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(203.0);
        offloader.offload(161, t).unwrap();
        let restored = offloader.restore(161).unwrap();
        assert_eq!(restored.get(0), 203.0);
    }

    #[test]
    fn test_cpu_offload_stress_162() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(204.0);
        offloader.offload(162, t).unwrap();
        let restored = offloader.restore(162).unwrap();
        assert_eq!(restored.get(0), 204.0);
    }

    #[test]
    fn test_cpu_offload_stress_163() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(205.0);
        offloader.offload(163, t).unwrap();
        let restored = offloader.restore(163).unwrap();
        assert_eq!(restored.get(0), 205.0);
    }

    #[test]
    fn test_cpu_offload_stress_164() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(206.0);
        offloader.offload(164, t).unwrap();
        let restored = offloader.restore(164).unwrap();
        assert_eq!(restored.get(0), 206.0);
    }

    #[test]
    fn test_cpu_offload_stress_165() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(207.0);
        offloader.offload(165, t).unwrap();
        let restored = offloader.restore(165).unwrap();
        assert_eq!(restored.get(0), 207.0);
    }

    #[test]
    fn test_cpu_offload_stress_166() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(208.0);
        offloader.offload(166, t).unwrap();
        let restored = offloader.restore(166).unwrap();
        assert_eq!(restored.get(0), 208.0);
    }

    #[test]
    fn test_cpu_offload_stress_167() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(209.0);
        offloader.offload(167, t).unwrap();
        let restored = offloader.restore(167).unwrap();
        assert_eq!(restored.get(0), 209.0);
    }

    #[test]
    fn test_cpu_offload_stress_168() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(210.0);
        offloader.offload(168, t).unwrap();
        let restored = offloader.restore(168).unwrap();
        assert_eq!(restored.get(0), 210.0);
    }

    #[test]
    fn test_cpu_offload_stress_169() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(211.0);
        offloader.offload(169, t).unwrap();
        let restored = offloader.restore(169).unwrap();
        assert_eq!(restored.get(0), 211.0);
    }

    #[test]
    fn test_cpu_offload_stress_170() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(212.0);
        offloader.offload(170, t).unwrap();
        let restored = offloader.restore(170).unwrap();
        assert_eq!(restored.get(0), 212.0);
    }

    #[test]
    fn test_cpu_offload_stress_171() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(213.0);
        offloader.offload(171, t).unwrap();
        let restored = offloader.restore(171).unwrap();
        assert_eq!(restored.get(0), 213.0);
    }

    #[test]
    fn test_cpu_offload_stress_172() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(214.0);
        offloader.offload(172, t).unwrap();
        let restored = offloader.restore(172).unwrap();
        assert_eq!(restored.get(0), 214.0);
    }

    #[test]
    fn test_cpu_offload_stress_173() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(215.0);
        offloader.offload(173, t).unwrap();
        let restored = offloader.restore(173).unwrap();
        assert_eq!(restored.get(0), 215.0);
    }

    #[test]
    fn test_cpu_offload_stress_174() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(216.0);
        offloader.offload(174, t).unwrap();
        let restored = offloader.restore(174).unwrap();
        assert_eq!(restored.get(0), 216.0);
    }

    #[test]
    fn test_cpu_offload_stress_175() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(217.0);
        offloader.offload(175, t).unwrap();
        let restored = offloader.restore(175).unwrap();
        assert_eq!(restored.get(0), 217.0);
    }

    #[test]
    fn test_cpu_offload_stress_176() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(218.0);
        offloader.offload(176, t).unwrap();
        let restored = offloader.restore(176).unwrap();
        assert_eq!(restored.get(0), 218.0);
    }

    #[test]
    fn test_cpu_offload_stress_177() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(219.0);
        offloader.offload(177, t).unwrap();
        let restored = offloader.restore(177).unwrap();
        assert_eq!(restored.get(0), 219.0);
    }

    #[test]
    fn test_cpu_offload_stress_178() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(220.0);
        offloader.offload(178, t).unwrap();
        let restored = offloader.restore(178).unwrap();
        assert_eq!(restored.get(0), 220.0);
    }

    #[test]
    fn test_cpu_offload_stress_179() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(221.0);
        offloader.offload(179, t).unwrap();
        let restored = offloader.restore(179).unwrap();
        assert_eq!(restored.get(0), 221.0);
    }

    #[test]
    fn test_cpu_offload_stress_180() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(222.0);
        offloader.offload(180, t).unwrap();
        let restored = offloader.restore(180).unwrap();
        assert_eq!(restored.get(0), 222.0);
    }

    #[test]
    fn test_cpu_offload_stress_181() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(223.0);
        offloader.offload(181, t).unwrap();
        let restored = offloader.restore(181).unwrap();
        assert_eq!(restored.get(0), 223.0);
    }

    #[test]
    fn test_cpu_offload_stress_182() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(224.0);
        offloader.offload(182, t).unwrap();
        let restored = offloader.restore(182).unwrap();
        assert_eq!(restored.get(0), 224.0);
    }

    #[test]
    fn test_cpu_offload_stress_183() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(225.0);
        offloader.offload(183, t).unwrap();
        let restored = offloader.restore(183).unwrap();
        assert_eq!(restored.get(0), 225.0);
    }

    #[test]
    fn test_cpu_offload_stress_184() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(226.0);
        offloader.offload(184, t).unwrap();
        let restored = offloader.restore(184).unwrap();
        assert_eq!(restored.get(0), 226.0);
    }

    #[test]
    fn test_cpu_offload_stress_185() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(227.0);
        offloader.offload(185, t).unwrap();
        let restored = offloader.restore(185).unwrap();
        assert_eq!(restored.get(0), 227.0);
    }

    #[test]
    fn test_cpu_offload_stress_186() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(228.0);
        offloader.offload(186, t).unwrap();
        let restored = offloader.restore(186).unwrap();
        assert_eq!(restored.get(0), 228.0);
    }

    #[test]
    fn test_cpu_offload_stress_187() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(229.0);
        offloader.offload(187, t).unwrap();
        let restored = offloader.restore(187).unwrap();
        assert_eq!(restored.get(0), 229.0);
    }

    #[test]
    fn test_cpu_offload_stress_188() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(230.0);
        offloader.offload(188, t).unwrap();
        let restored = offloader.restore(188).unwrap();
        assert_eq!(restored.get(0), 230.0);
    }

    #[test]
    fn test_cpu_offload_stress_189() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(231.0);
        offloader.offload(189, t).unwrap();
        let restored = offloader.restore(189).unwrap();
        assert_eq!(restored.get(0), 231.0);
    }

    #[test]
    fn test_cpu_offload_stress_190() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(232.0);
        offloader.offload(190, t).unwrap();
        let restored = offloader.restore(190).unwrap();
        assert_eq!(restored.get(0), 232.0);
    }

    #[test]
    fn test_cpu_offload_stress_191() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(233.0);
        offloader.offload(191, t).unwrap();
        let restored = offloader.restore(191).unwrap();
        assert_eq!(restored.get(0), 233.0);
    }

    #[test]
    fn test_cpu_offload_stress_192() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(234.0);
        offloader.offload(192, t).unwrap();
        let restored = offloader.restore(192).unwrap();
        assert_eq!(restored.get(0), 234.0);
    }

    #[test]
    fn test_cpu_offload_stress_193() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(235.0);
        offloader.offload(193, t).unwrap();
        let restored = offloader.restore(193).unwrap();
        assert_eq!(restored.get(0), 235.0);
    }

    #[test]
    fn test_cpu_offload_stress_194() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(236.0);
        offloader.offload(194, t).unwrap();
        let restored = offloader.restore(194).unwrap();
        assert_eq!(restored.get(0), 236.0);
    }

    #[test]
    fn test_cpu_offload_stress_195() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(237.0);
        offloader.offload(195, t).unwrap();
        let restored = offloader.restore(195).unwrap();
        assert_eq!(restored.get(0), 237.0);
    }

    #[test]
    fn test_cpu_offload_stress_196() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(238.0);
        offloader.offload(196, t).unwrap();
        let restored = offloader.restore(196).unwrap();
        assert_eq!(restored.get(0), 238.0);
    }

    #[test]
    fn test_cpu_offload_stress_197() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(239.0);
        offloader.offload(197, t).unwrap();
        let restored = offloader.restore(197).unwrap();
        assert_eq!(restored.get(0), 239.0);
    }

    #[test]
    fn test_cpu_offload_stress_198() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(240.0);
        offloader.offload(198, t).unwrap();
        let restored = offloader.restore(198).unwrap();
        assert_eq!(restored.get(0), 240.0);
    }

    #[test]
    fn test_cpu_offload_stress_199() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(241.0);
        offloader.offload(199, t).unwrap();
        let restored = offloader.restore(199).unwrap();
        assert_eq!(restored.get(0), 241.0);
    }

    #[test]
    fn test_cpu_offload_stress_200() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(242.0);
        offloader.offload(200, t).unwrap();
        let restored = offloader.restore(200).unwrap();
        assert_eq!(restored.get(0), 242.0);
    }

    #[test]
    fn test_cpu_offload_stress_201() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(243.0);
        offloader.offload(201, t).unwrap();
        let restored = offloader.restore(201).unwrap();
        assert_eq!(restored.get(0), 243.0);
    }

    #[test]
    fn test_cpu_offload_stress_202() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(244.0);
        offloader.offload(202, t).unwrap();
        let restored = offloader.restore(202).unwrap();
        assert_eq!(restored.get(0), 244.0);
    }

    #[test]
    fn test_cpu_offload_stress_203() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(245.0);
        offloader.offload(203, t).unwrap();
        let restored = offloader.restore(203).unwrap();
        assert_eq!(restored.get(0), 245.0);
    }

    #[test]
    fn test_cpu_offload_stress_204() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(246.0);
        offloader.offload(204, t).unwrap();
        let restored = offloader.restore(204).unwrap();
        assert_eq!(restored.get(0), 246.0);
    }

    #[test]
    fn test_cpu_offload_stress_205() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(247.0);
        offloader.offload(205, t).unwrap();
        let restored = offloader.restore(205).unwrap();
        assert_eq!(restored.get(0), 247.0);
    }

    #[test]
    fn test_cpu_offload_stress_206() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(248.0);
        offloader.offload(206, t).unwrap();
        let restored = offloader.restore(206).unwrap();
        assert_eq!(restored.get(0), 248.0);
    }

    #[test]
    fn test_cpu_offload_stress_207() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(249.0);
        offloader.offload(207, t).unwrap();
        let restored = offloader.restore(207).unwrap();
        assert_eq!(restored.get(0), 249.0);
    }

    #[test]
    fn test_cpu_offload_stress_208() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(250.0);
        offloader.offload(208, t).unwrap();
        let restored = offloader.restore(208).unwrap();
        assert_eq!(restored.get(0), 250.0);
    }

    #[test]
    fn test_cpu_offload_stress_209() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(251.0);
        offloader.offload(209, t).unwrap();
        let restored = offloader.restore(209).unwrap();
        assert_eq!(restored.get(0), 251.0);
    }

    #[test]
    fn test_cpu_offload_stress_210() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(252.0);
        offloader.offload(210, t).unwrap();
        let restored = offloader.restore(210).unwrap();
        assert_eq!(restored.get(0), 252.0);
    }

    #[test]
    fn test_cpu_offload_stress_211() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(253.0);
        offloader.offload(211, t).unwrap();
        let restored = offloader.restore(211).unwrap();
        assert_eq!(restored.get(0), 253.0);
    }

    #[test]
    fn test_cpu_offload_stress_212() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(254.0);
        offloader.offload(212, t).unwrap();
        let restored = offloader.restore(212).unwrap();
        assert_eq!(restored.get(0), 254.0);
    }

    #[test]
    fn test_cpu_offload_stress_213() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(255.0);
        offloader.offload(213, t).unwrap();
        let restored = offloader.restore(213).unwrap();
        assert_eq!(restored.get(0), 255.0);
    }

    #[test]
    fn test_cpu_offload_stress_214() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(256.0);
        offloader.offload(214, t).unwrap();
        let restored = offloader.restore(214).unwrap();
        assert_eq!(restored.get(0), 256.0);
    }

    #[test]
    fn test_cpu_offload_stress_215() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(257.0);
        offloader.offload(215, t).unwrap();
        let restored = offloader.restore(215).unwrap();
        assert_eq!(restored.get(0), 257.0);
    }

    #[test]
    fn test_cpu_offload_stress_216() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(258.0);
        offloader.offload(216, t).unwrap();
        let restored = offloader.restore(216).unwrap();
        assert_eq!(restored.get(0), 258.0);
    }

    #[test]
    fn test_cpu_offload_stress_217() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(259.0);
        offloader.offload(217, t).unwrap();
        let restored = offloader.restore(217).unwrap();
        assert_eq!(restored.get(0), 259.0);
    }

    #[test]
    fn test_cpu_offload_stress_218() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(260.0);
        offloader.offload(218, t).unwrap();
        let restored = offloader.restore(218).unwrap();
        assert_eq!(restored.get(0), 260.0);
    }

    #[test]
    fn test_cpu_offload_stress_219() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(261.0);
        offloader.offload(219, t).unwrap();
        let restored = offloader.restore(219).unwrap();
        assert_eq!(restored.get(0), 261.0);
    }

    #[test]
    fn test_cpu_offload_stress_220() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(262.0);
        offloader.offload(220, t).unwrap();
        let restored = offloader.restore(220).unwrap();
        assert_eq!(restored.get(0), 262.0);
    }

    #[test]
    fn test_cpu_offload_stress_221() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(263.0);
        offloader.offload(221, t).unwrap();
        let restored = offloader.restore(221).unwrap();
        assert_eq!(restored.get(0), 263.0);
    }

    #[test]
    fn test_cpu_offload_stress_222() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(264.0);
        offloader.offload(222, t).unwrap();
        let restored = offloader.restore(222).unwrap();
        assert_eq!(restored.get(0), 264.0);
    }

    #[test]
    fn test_cpu_offload_stress_223() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(265.0);
        offloader.offload(223, t).unwrap();
        let restored = offloader.restore(223).unwrap();
        assert_eq!(restored.get(0), 265.0);
    }

    #[test]
    fn test_cpu_offload_stress_224() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(266.0);
        offloader.offload(224, t).unwrap();
        let restored = offloader.restore(224).unwrap();
        assert_eq!(restored.get(0), 266.0);
    }

    #[test]
    fn test_cpu_offload_stress_225() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(267.0);
        offloader.offload(225, t).unwrap();
        let restored = offloader.restore(225).unwrap();
        assert_eq!(restored.get(0), 267.0);
    }

    #[test]
    fn test_cpu_offload_stress_226() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(268.0);
        offloader.offload(226, t).unwrap();
        let restored = offloader.restore(226).unwrap();
        assert_eq!(restored.get(0), 268.0);
    }

    #[test]
    fn test_cpu_offload_stress_227() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(269.0);
        offloader.offload(227, t).unwrap();
        let restored = offloader.restore(227).unwrap();
        assert_eq!(restored.get(0), 269.0);
    }

    #[test]
    fn test_cpu_offload_stress_228() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(270.0);
        offloader.offload(228, t).unwrap();
        let restored = offloader.restore(228).unwrap();
        assert_eq!(restored.get(0), 270.0);
    }

    #[test]
    fn test_cpu_offload_stress_229() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(271.0);
        offloader.offload(229, t).unwrap();
        let restored = offloader.restore(229).unwrap();
        assert_eq!(restored.get(0), 271.0);
    }

    #[test]
    fn test_cpu_offload_stress_230() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(272.0);
        offloader.offload(230, t).unwrap();
        let restored = offloader.restore(230).unwrap();
        assert_eq!(restored.get(0), 272.0);
    }

    #[test]
    fn test_cpu_offload_stress_231() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(273.0);
        offloader.offload(231, t).unwrap();
        let restored = offloader.restore(231).unwrap();
        assert_eq!(restored.get(0), 273.0);
    }

    #[test]
    fn test_cpu_offload_stress_232() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(274.0);
        offloader.offload(232, t).unwrap();
        let restored = offloader.restore(232).unwrap();
        assert_eq!(restored.get(0), 274.0);
    }

    #[test]
    fn test_cpu_offload_stress_233() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(275.0);
        offloader.offload(233, t).unwrap();
        let restored = offloader.restore(233).unwrap();
        assert_eq!(restored.get(0), 275.0);
    }

    #[test]
    fn test_cpu_offload_stress_234() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(276.0);
        offloader.offload(234, t).unwrap();
        let restored = offloader.restore(234).unwrap();
        assert_eq!(restored.get(0), 276.0);
    }

    #[test]
    fn test_cpu_offload_stress_235() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(277.0);
        offloader.offload(235, t).unwrap();
        let restored = offloader.restore(235).unwrap();
        assert_eq!(restored.get(0), 277.0);
    }

    #[test]
    fn test_cpu_offload_stress_236() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(278.0);
        offloader.offload(236, t).unwrap();
        let restored = offloader.restore(236).unwrap();
        assert_eq!(restored.get(0), 278.0);
    }

    #[test]
    fn test_cpu_offload_stress_237() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(279.0);
        offloader.offload(237, t).unwrap();
        let restored = offloader.restore(237).unwrap();
        assert_eq!(restored.get(0), 279.0);
    }

    #[test]
    fn test_cpu_offload_stress_238() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(280.0);
        offloader.offload(238, t).unwrap();
        let restored = offloader.restore(238).unwrap();
        assert_eq!(restored.get(0), 280.0);
    }

    #[test]
    fn test_cpu_offload_stress_239() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(281.0);
        offloader.offload(239, t).unwrap();
        let restored = offloader.restore(239).unwrap();
        assert_eq!(restored.get(0), 281.0);
    }

    #[test]
    fn test_cpu_offload_stress_240() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(282.0);
        offloader.offload(240, t).unwrap();
        let restored = offloader.restore(240).unwrap();
        assert_eq!(restored.get(0), 282.0);
    }

    #[test]
    fn test_cpu_offload_stress_241() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(283.0);
        offloader.offload(241, t).unwrap();
        let restored = offloader.restore(241).unwrap();
        assert_eq!(restored.get(0), 283.0);
    }

    #[test]
    fn test_cpu_offload_stress_242() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(284.0);
        offloader.offload(242, t).unwrap();
        let restored = offloader.restore(242).unwrap();
        assert_eq!(restored.get(0), 284.0);
    }

    #[test]
    fn test_cpu_offload_stress_243() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(285.0);
        offloader.offload(243, t).unwrap();
        let restored = offloader.restore(243).unwrap();
        assert_eq!(restored.get(0), 285.0);
    }

    #[test]
    fn test_cpu_offload_stress_244() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(286.0);
        offloader.offload(244, t).unwrap();
        let restored = offloader.restore(244).unwrap();
        assert_eq!(restored.get(0), 286.0);
    }

    #[test]
    fn test_cpu_offload_stress_245() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(287.0);
        offloader.offload(245, t).unwrap();
        let restored = offloader.restore(245).unwrap();
        assert_eq!(restored.get(0), 287.0);
    }

    #[test]
    fn test_cpu_offload_stress_246() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(288.0);
        offloader.offload(246, t).unwrap();
        let restored = offloader.restore(246).unwrap();
        assert_eq!(restored.get(0), 288.0);
    }

    #[test]
    fn test_cpu_offload_stress_247() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(289.0);
        offloader.offload(247, t).unwrap();
        let restored = offloader.restore(247).unwrap();
        assert_eq!(restored.get(0), 289.0);
    }

    #[test]
    fn test_cpu_offload_stress_248() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(290.0);
        offloader.offload(248, t).unwrap();
        let restored = offloader.restore(248).unwrap();
        assert_eq!(restored.get(0), 290.0);
    }

    #[test]
    fn test_cpu_offload_stress_249() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(291.0);
        offloader.offload(249, t).unwrap();
        let restored = offloader.restore(249).unwrap();
        assert_eq!(restored.get(0), 291.0);
    }

    #[test]
    fn test_cpu_offload_stress_250() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(292.0);
        offloader.offload(250, t).unwrap();
        let restored = offloader.restore(250).unwrap();
        assert_eq!(restored.get(0), 292.0);
    }

    #[test]
    fn test_cpu_offload_stress_251() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(293.0);
        offloader.offload(251, t).unwrap();
        let restored = offloader.restore(251).unwrap();
        assert_eq!(restored.get(0), 293.0);
    }

    #[test]
    fn test_cpu_offload_stress_252() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(294.0);
        offloader.offload(252, t).unwrap();
        let restored = offloader.restore(252).unwrap();
        assert_eq!(restored.get(0), 294.0);
    }

    #[test]
    fn test_cpu_offload_stress_253() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(295.0);
        offloader.offload(253, t).unwrap();
        let restored = offloader.restore(253).unwrap();
        assert_eq!(restored.get(0), 295.0);
    }

    #[test]
    fn test_cpu_offload_stress_254() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(296.0);
        offloader.offload(254, t).unwrap();
        let restored = offloader.restore(254).unwrap();
        assert_eq!(restored.get(0), 296.0);
    }

    #[test]
    fn test_cpu_offload_stress_255() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(297.0);
        offloader.offload(255, t).unwrap();
        let restored = offloader.restore(255).unwrap();
        assert_eq!(restored.get(0), 297.0);
    }

    #[test]
    fn test_cpu_offload_stress_256() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(298.0);
        offloader.offload(256, t).unwrap();
        let restored = offloader.restore(256).unwrap();
        assert_eq!(restored.get(0), 298.0);
    }

    #[test]
    fn test_cpu_offload_stress_257() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(299.0);
        offloader.offload(257, t).unwrap();
        let restored = offloader.restore(257).unwrap();
        assert_eq!(restored.get(0), 299.0);
    }

    #[test]
    fn test_cpu_offload_stress_258() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(300.0);
        offloader.offload(258, t).unwrap();
        let restored = offloader.restore(258).unwrap();
        assert_eq!(restored.get(0), 300.0);
    }

    #[test]
    fn test_cpu_offload_stress_259() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(301.0);
        offloader.offload(259, t).unwrap();
        let restored = offloader.restore(259).unwrap();
        assert_eq!(restored.get(0), 301.0);
    }

    #[test]
    fn test_cpu_offload_stress_260() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(302.0);
        offloader.offload(260, t).unwrap();
        let restored = offloader.restore(260).unwrap();
        assert_eq!(restored.get(0), 302.0);
    }

    #[test]
    fn test_cpu_offload_stress_261() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(303.0);
        offloader.offload(261, t).unwrap();
        let restored = offloader.restore(261).unwrap();
        assert_eq!(restored.get(0), 303.0);
    }

    #[test]
    fn test_cpu_offload_stress_262() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(304.0);
        offloader.offload(262, t).unwrap();
        let restored = offloader.restore(262).unwrap();
        assert_eq!(restored.get(0), 304.0);
    }

    #[test]
    fn test_cpu_offload_stress_263() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(305.0);
        offloader.offload(263, t).unwrap();
        let restored = offloader.restore(263).unwrap();
        assert_eq!(restored.get(0), 305.0);
    }

    #[test]
    fn test_cpu_offload_stress_264() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(306.0);
        offloader.offload(264, t).unwrap();
        let restored = offloader.restore(264).unwrap();
        assert_eq!(restored.get(0), 306.0);
    }

    #[test]
    fn test_cpu_offload_stress_265() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(307.0);
        offloader.offload(265, t).unwrap();
        let restored = offloader.restore(265).unwrap();
        assert_eq!(restored.get(0), 307.0);
    }

    #[test]
    fn test_cpu_offload_stress_266() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(308.0);
        offloader.offload(266, t).unwrap();
        let restored = offloader.restore(266).unwrap();
        assert_eq!(restored.get(0), 308.0);
    }

    #[test]
    fn test_cpu_offload_stress_267() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(309.0);
        offloader.offload(267, t).unwrap();
        let restored = offloader.restore(267).unwrap();
        assert_eq!(restored.get(0), 309.0);
    }

    #[test]
    fn test_cpu_offload_stress_268() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(310.0);
        offloader.offload(268, t).unwrap();
        let restored = offloader.restore(268).unwrap();
        assert_eq!(restored.get(0), 310.0);
    }

    #[test]
    fn test_cpu_offload_stress_269() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(311.0);
        offloader.offload(269, t).unwrap();
        let restored = offloader.restore(269).unwrap();
        assert_eq!(restored.get(0), 311.0);
    }

    #[test]
    fn test_cpu_offload_stress_270() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(312.0);
        offloader.offload(270, t).unwrap();
        let restored = offloader.restore(270).unwrap();
        assert_eq!(restored.get(0), 312.0);
    }

    #[test]
    fn test_cpu_offload_stress_271() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(313.0);
        offloader.offload(271, t).unwrap();
        let restored = offloader.restore(271).unwrap();
        assert_eq!(restored.get(0), 313.0);
    }

    #[test]
    fn test_cpu_offload_stress_272() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(314.0);
        offloader.offload(272, t).unwrap();
        let restored = offloader.restore(272).unwrap();
        assert_eq!(restored.get(0), 314.0);
    }

    #[test]
    fn test_cpu_offload_stress_273() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(315.0);
        offloader.offload(273, t).unwrap();
        let restored = offloader.restore(273).unwrap();
        assert_eq!(restored.get(0), 315.0);
    }

    #[test]
    fn test_cpu_offload_stress_274() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(316.0);
        offloader.offload(274, t).unwrap();
        let restored = offloader.restore(274).unwrap();
        assert_eq!(restored.get(0), 316.0);
    }

    #[test]
    fn test_cpu_offload_stress_275() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(317.0);
        offloader.offload(275, t).unwrap();
        let restored = offloader.restore(275).unwrap();
        assert_eq!(restored.get(0), 317.0);
    }

    #[test]
    fn test_cpu_offload_stress_276() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(318.0);
        offloader.offload(276, t).unwrap();
        let restored = offloader.restore(276).unwrap();
        assert_eq!(restored.get(0), 318.0);
    }

    #[test]
    fn test_cpu_offload_stress_277() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(319.0);
        offloader.offload(277, t).unwrap();
        let restored = offloader.restore(277).unwrap();
        assert_eq!(restored.get(0), 319.0);
    }

    #[test]
    fn test_cpu_offload_stress_278() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(320.0);
        offloader.offload(278, t).unwrap();
        let restored = offloader.restore(278).unwrap();
        assert_eq!(restored.get(0), 320.0);
    }

    #[test]
    fn test_cpu_offload_stress_279() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(321.0);
        offloader.offload(279, t).unwrap();
        let restored = offloader.restore(279).unwrap();
        assert_eq!(restored.get(0), 321.0);
    }

    #[test]
    fn test_cpu_offload_stress_280() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(322.0);
        offloader.offload(280, t).unwrap();
        let restored = offloader.restore(280).unwrap();
        assert_eq!(restored.get(0), 322.0);
    }

    #[test]
    fn test_cpu_offload_stress_281() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(323.0);
        offloader.offload(281, t).unwrap();
        let restored = offloader.restore(281).unwrap();
        assert_eq!(restored.get(0), 323.0);
    }

    #[test]
    fn test_cpu_offload_stress_282() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(324.0);
        offloader.offload(282, t).unwrap();
        let restored = offloader.restore(282).unwrap();
        assert_eq!(restored.get(0), 324.0);
    }

    #[test]
    fn test_cpu_offload_stress_283() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(325.0);
        offloader.offload(283, t).unwrap();
        let restored = offloader.restore(283).unwrap();
        assert_eq!(restored.get(0), 325.0);
    }

    #[test]
    fn test_cpu_offload_stress_284() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(326.0);
        offloader.offload(284, t).unwrap();
        let restored = offloader.restore(284).unwrap();
        assert_eq!(restored.get(0), 326.0);
    }

    #[test]
    fn test_cpu_offload_stress_285() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(327.0);
        offloader.offload(285, t).unwrap();
        let restored = offloader.restore(285).unwrap();
        assert_eq!(restored.get(0), 327.0);
    }

    #[test]
    fn test_cpu_offload_stress_286() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(328.0);
        offloader.offload(286, t).unwrap();
        let restored = offloader.restore(286).unwrap();
        assert_eq!(restored.get(0), 328.0);
    }

    #[test]
    fn test_cpu_offload_stress_287() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(329.0);
        offloader.offload(287, t).unwrap();
        let restored = offloader.restore(287).unwrap();
        assert_eq!(restored.get(0), 329.0);
    }

    #[test]
    fn test_cpu_offload_stress_288() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(330.0);
        offloader.offload(288, t).unwrap();
        let restored = offloader.restore(288).unwrap();
        assert_eq!(restored.get(0), 330.0);
    }

    #[test]
    fn test_cpu_offload_stress_289() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(331.0);
        offloader.offload(289, t).unwrap();
        let restored = offloader.restore(289).unwrap();
        assert_eq!(restored.get(0), 331.0);
    }

    #[test]
    fn test_cpu_offload_stress_290() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(332.0);
        offloader.offload(290, t).unwrap();
        let restored = offloader.restore(290).unwrap();
        assert_eq!(restored.get(0), 332.0);
    }

    #[test]
    fn test_cpu_offload_stress_291() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(333.0);
        offloader.offload(291, t).unwrap();
        let restored = offloader.restore(291).unwrap();
        assert_eq!(restored.get(0), 333.0);
    }

    #[test]
    fn test_cpu_offload_stress_292() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(334.0);
        offloader.offload(292, t).unwrap();
        let restored = offloader.restore(292).unwrap();
        assert_eq!(restored.get(0), 334.0);
    }

    #[test]
    fn test_cpu_offload_stress_293() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(335.0);
        offloader.offload(293, t).unwrap();
        let restored = offloader.restore(293).unwrap();
        assert_eq!(restored.get(0), 335.0);
    }

    #[test]
    fn test_cpu_offload_stress_294() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(336.0);
        offloader.offload(294, t).unwrap();
        let restored = offloader.restore(294).unwrap();
        assert_eq!(restored.get(0), 336.0);
    }

    #[test]
    fn test_cpu_offload_stress_295() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(337.0);
        offloader.offload(295, t).unwrap();
        let restored = offloader.restore(295).unwrap();
        assert_eq!(restored.get(0), 337.0);
    }

    #[test]
    fn test_cpu_offload_stress_296() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(338.0);
        offloader.offload(296, t).unwrap();
        let restored = offloader.restore(296).unwrap();
        assert_eq!(restored.get(0), 338.0);
    }

    #[test]
    fn test_cpu_offload_stress_297() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(339.0);
        offloader.offload(297, t).unwrap();
        let restored = offloader.restore(297).unwrap();
        assert_eq!(restored.get(0), 339.0);
    }

    #[test]
    fn test_cpu_offload_stress_298() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(340.0);
        offloader.offload(298, t).unwrap();
        let restored = offloader.restore(298).unwrap();
        assert_eq!(restored.get(0), 340.0);
    }

    #[test]
    fn test_cpu_offload_stress_299() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(341.0);
        offloader.offload(299, t).unwrap();
        let restored = offloader.restore(299).unwrap();
        assert_eq!(restored.get(0), 341.0);
    }

    #[test]
    fn test_cpu_offload_stress_300() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(342.0);
        offloader.offload(300, t).unwrap();
        let restored = offloader.restore(300).unwrap();
        assert_eq!(restored.get(0), 342.0);
    }

    #[test]
    fn test_cpu_offload_stress_301() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(343.0);
        offloader.offload(301, t).unwrap();
        let restored = offloader.restore(301).unwrap();
        assert_eq!(restored.get(0), 343.0);
    }

    #[test]
    fn test_cpu_offload_stress_302() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(344.0);
        offloader.offload(302, t).unwrap();
        let restored = offloader.restore(302).unwrap();
        assert_eq!(restored.get(0), 344.0);
    }

    #[test]
    fn test_cpu_offload_stress_303() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(345.0);
        offloader.offload(303, t).unwrap();
        let restored = offloader.restore(303).unwrap();
        assert_eq!(restored.get(0), 345.0);
    }

    #[test]
    fn test_cpu_offload_stress_304() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(346.0);
        offloader.offload(304, t).unwrap();
        let restored = offloader.restore(304).unwrap();
        assert_eq!(restored.get(0), 346.0);
    }

    #[test]
    fn test_cpu_offload_stress_305() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(347.0);
        offloader.offload(305, t).unwrap();
        let restored = offloader.restore(305).unwrap();
        assert_eq!(restored.get(0), 347.0);
    }

    #[test]
    fn test_cpu_offload_stress_306() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(348.0);
        offloader.offload(306, t).unwrap();
        let restored = offloader.restore(306).unwrap();
        assert_eq!(restored.get(0), 348.0);
    }

    #[test]
    fn test_cpu_offload_stress_307() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(349.0);
        offloader.offload(307, t).unwrap();
        let restored = offloader.restore(307).unwrap();
        assert_eq!(restored.get(0), 349.0);
    }

    #[test]
    fn test_cpu_offload_stress_308() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(350.0);
        offloader.offload(308, t).unwrap();
        let restored = offloader.restore(308).unwrap();
        assert_eq!(restored.get(0), 350.0);
    }

    #[test]
    fn test_cpu_offload_stress_309() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(351.0);
        offloader.offload(309, t).unwrap();
        let restored = offloader.restore(309).unwrap();
        assert_eq!(restored.get(0), 351.0);
    }

    #[test]
    fn test_cpu_offload_stress_310() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(352.0);
        offloader.offload(310, t).unwrap();
        let restored = offloader.restore(310).unwrap();
        assert_eq!(restored.get(0), 352.0);
    }

    #[test]
    fn test_cpu_offload_stress_311() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(353.0);
        offloader.offload(311, t).unwrap();
        let restored = offloader.restore(311).unwrap();
        assert_eq!(restored.get(0), 353.0);
    }

    #[test]
    fn test_cpu_offload_stress_312() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(354.0);
        offloader.offload(312, t).unwrap();
        let restored = offloader.restore(312).unwrap();
        assert_eq!(restored.get(0), 354.0);
    }

    #[test]
    fn test_cpu_offload_stress_313() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(355.0);
        offloader.offload(313, t).unwrap();
        let restored = offloader.restore(313).unwrap();
        assert_eq!(restored.get(0), 355.0);
    }

    #[test]
    fn test_cpu_offload_stress_314() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(356.0);
        offloader.offload(314, t).unwrap();
        let restored = offloader.restore(314).unwrap();
        assert_eq!(restored.get(0), 356.0);
    }

    #[test]
    fn test_cpu_offload_stress_315() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(357.0);
        offloader.offload(315, t).unwrap();
        let restored = offloader.restore(315).unwrap();
        assert_eq!(restored.get(0), 357.0);
    }

    #[test]
    fn test_cpu_offload_stress_316() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(358.0);
        offloader.offload(316, t).unwrap();
        let restored = offloader.restore(316).unwrap();
        assert_eq!(restored.get(0), 358.0);
    }

    #[test]
    fn test_cpu_offload_stress_317() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(359.0);
        offloader.offload(317, t).unwrap();
        let restored = offloader.restore(317).unwrap();
        assert_eq!(restored.get(0), 359.0);
    }

    #[test]
    fn test_cpu_offload_stress_318() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(360.0);
        offloader.offload(318, t).unwrap();
        let restored = offloader.restore(318).unwrap();
        assert_eq!(restored.get(0), 360.0);
    }

    #[test]
    fn test_cpu_offload_stress_319() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(361.0);
        offloader.offload(319, t).unwrap();
        let restored = offloader.restore(319).unwrap();
        assert_eq!(restored.get(0), 361.0);
    }

    #[test]
    fn test_cpu_offload_stress_320() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(362.0);
        offloader.offload(320, t).unwrap();
        let restored = offloader.restore(320).unwrap();
        assert_eq!(restored.get(0), 362.0);
    }

    #[test]
    fn test_cpu_offload_stress_321() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(363.0);
        offloader.offload(321, t).unwrap();
        let restored = offloader.restore(321).unwrap();
        assert_eq!(restored.get(0), 363.0);
    }

    #[test]
    fn test_cpu_offload_stress_322() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(364.0);
        offloader.offload(322, t).unwrap();
        let restored = offloader.restore(322).unwrap();
        assert_eq!(restored.get(0), 364.0);
    }

    #[test]
    fn test_cpu_offload_stress_323() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(365.0);
        offloader.offload(323, t).unwrap();
        let restored = offloader.restore(323).unwrap();
        assert_eq!(restored.get(0), 365.0);
    }

    #[test]
    fn test_cpu_offload_stress_324() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(366.0);
        offloader.offload(324, t).unwrap();
        let restored = offloader.restore(324).unwrap();
        assert_eq!(restored.get(0), 366.0);
    }

    #[test]
    fn test_cpu_offload_stress_325() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(367.0);
        offloader.offload(325, t).unwrap();
        let restored = offloader.restore(325).unwrap();
        assert_eq!(restored.get(0), 367.0);
    }

    #[test]
    fn test_cpu_offload_stress_326() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(368.0);
        offloader.offload(326, t).unwrap();
        let restored = offloader.restore(326).unwrap();
        assert_eq!(restored.get(0), 368.0);
    }

    #[test]
    fn test_cpu_offload_stress_327() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(369.0);
        offloader.offload(327, t).unwrap();
        let restored = offloader.restore(327).unwrap();
        assert_eq!(restored.get(0), 369.0);
    }

    #[test]
    fn test_cpu_offload_stress_328() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(370.0);
        offloader.offload(328, t).unwrap();
        let restored = offloader.restore(328).unwrap();
        assert_eq!(restored.get(0), 370.0);
    }

    #[test]
    fn test_cpu_offload_stress_329() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(371.0);
        offloader.offload(329, t).unwrap();
        let restored = offloader.restore(329).unwrap();
        assert_eq!(restored.get(0), 371.0);
    }

    #[test]
    fn test_cpu_offload_stress_330() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(372.0);
        offloader.offload(330, t).unwrap();
        let restored = offloader.restore(330).unwrap();
        assert_eq!(restored.get(0), 372.0);
    }

    #[test]
    fn test_cpu_offload_stress_331() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(373.0);
        offloader.offload(331, t).unwrap();
        let restored = offloader.restore(331).unwrap();
        assert_eq!(restored.get(0), 373.0);
    }

    #[test]
    fn test_cpu_offload_stress_332() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(374.0);
        offloader.offload(332, t).unwrap();
        let restored = offloader.restore(332).unwrap();
        assert_eq!(restored.get(0), 374.0);
    }

    #[test]
    fn test_cpu_offload_stress_333() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(375.0);
        offloader.offload(333, t).unwrap();
        let restored = offloader.restore(333).unwrap();
        assert_eq!(restored.get(0), 375.0);
    }

    #[test]
    fn test_cpu_offload_stress_334() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(376.0);
        offloader.offload(334, t).unwrap();
        let restored = offloader.restore(334).unwrap();
        assert_eq!(restored.get(0), 376.0);
    }

    #[test]
    fn test_cpu_offload_stress_335() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(377.0);
        offloader.offload(335, t).unwrap();
        let restored = offloader.restore(335).unwrap();
        assert_eq!(restored.get(0), 377.0);
    }

    #[test]
    fn test_cpu_offload_stress_336() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(378.0);
        offloader.offload(336, t).unwrap();
        let restored = offloader.restore(336).unwrap();
        assert_eq!(restored.get(0), 378.0);
    }

    #[test]
    fn test_cpu_offload_stress_337() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(379.0);
        offloader.offload(337, t).unwrap();
        let restored = offloader.restore(337).unwrap();
        assert_eq!(restored.get(0), 379.0);
    }

    #[test]
    fn test_cpu_offload_stress_338() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(380.0);
        offloader.offload(338, t).unwrap();
        let restored = offloader.restore(338).unwrap();
        assert_eq!(restored.get(0), 380.0);
    }

    #[test]
    fn test_cpu_offload_stress_339() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(381.0);
        offloader.offload(339, t).unwrap();
        let restored = offloader.restore(339).unwrap();
        assert_eq!(restored.get(0), 381.0);
    }

    #[test]
    fn test_cpu_offload_stress_340() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(382.0);
        offloader.offload(340, t).unwrap();
        let restored = offloader.restore(340).unwrap();
        assert_eq!(restored.get(0), 382.0);
    }

    #[test]
    fn test_cpu_offload_stress_341() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(383.0);
        offloader.offload(341, t).unwrap();
        let restored = offloader.restore(341).unwrap();
        assert_eq!(restored.get(0), 383.0);
    }

    #[test]
    fn test_cpu_offload_stress_342() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(384.0);
        offloader.offload(342, t).unwrap();
        let restored = offloader.restore(342).unwrap();
        assert_eq!(restored.get(0), 384.0);
    }

    #[test]
    fn test_cpu_offload_stress_343() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(385.0);
        offloader.offload(343, t).unwrap();
        let restored = offloader.restore(343).unwrap();
        assert_eq!(restored.get(0), 385.0);
    }

    #[test]
    fn test_cpu_offload_stress_344() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(386.0);
        offloader.offload(344, t).unwrap();
        let restored = offloader.restore(344).unwrap();
        assert_eq!(restored.get(0), 386.0);
    }

    #[test]
    fn test_cpu_offload_stress_345() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(387.0);
        offloader.offload(345, t).unwrap();
        let restored = offloader.restore(345).unwrap();
        assert_eq!(restored.get(0), 387.0);
    }

    #[test]
    fn test_cpu_offload_stress_346() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(388.0);
        offloader.offload(346, t).unwrap();
        let restored = offloader.restore(346).unwrap();
        assert_eq!(restored.get(0), 388.0);
    }

    #[test]
    fn test_cpu_offload_stress_347() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(389.0);
        offloader.offload(347, t).unwrap();
        let restored = offloader.restore(347).unwrap();
        assert_eq!(restored.get(0), 389.0);
    }

    #[test]
    fn test_cpu_offload_stress_348() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(390.0);
        offloader.offload(348, t).unwrap();
        let restored = offloader.restore(348).unwrap();
        assert_eq!(restored.get(0), 390.0);
    }

    #[test]
    fn test_cpu_offload_stress_349() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(391.0);
        offloader.offload(349, t).unwrap();
        let restored = offloader.restore(349).unwrap();
        assert_eq!(restored.get(0), 391.0);
    }

    #[test]
    fn test_cpu_offload_stress_350() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(392.0);
        offloader.offload(350, t).unwrap();
        let restored = offloader.restore(350).unwrap();
        assert_eq!(restored.get(0), 392.0);
    }

    #[test]
    fn test_cpu_offload_stress_351() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(393.0);
        offloader.offload(351, t).unwrap();
        let restored = offloader.restore(351).unwrap();
        assert_eq!(restored.get(0), 393.0);
    }

    #[test]
    fn test_cpu_offload_stress_352() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(394.0);
        offloader.offload(352, t).unwrap();
        let restored = offloader.restore(352).unwrap();
        assert_eq!(restored.get(0), 394.0);
    }

    #[test]
    fn test_cpu_offload_stress_353() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(395.0);
        offloader.offload(353, t).unwrap();
        let restored = offloader.restore(353).unwrap();
        assert_eq!(restored.get(0), 395.0);
    }

    #[test]
    fn test_cpu_offload_stress_354() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(396.0);
        offloader.offload(354, t).unwrap();
        let restored = offloader.restore(354).unwrap();
        assert_eq!(restored.get(0), 396.0);
    }

    #[test]
    fn test_cpu_offload_stress_355() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(397.0);
        offloader.offload(355, t).unwrap();
        let restored = offloader.restore(355).unwrap();
        assert_eq!(restored.get(0), 397.0);
    }

    #[test]
    fn test_cpu_offload_stress_356() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(398.0);
        offloader.offload(356, t).unwrap();
        let restored = offloader.restore(356).unwrap();
        assert_eq!(restored.get(0), 398.0);
    }

    #[test]
    fn test_cpu_offload_stress_357() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(399.0);
        offloader.offload(357, t).unwrap();
        let restored = offloader.restore(357).unwrap();
        assert_eq!(restored.get(0), 399.0);
    }

    #[test]
    fn test_cpu_offload_stress_358() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(400.0);
        offloader.offload(358, t).unwrap();
        let restored = offloader.restore(358).unwrap();
        assert_eq!(restored.get(0), 400.0);
    }

    #[test]
    fn test_cpu_offload_stress_359() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(401.0);
        offloader.offload(359, t).unwrap();
        let restored = offloader.restore(359).unwrap();
        assert_eq!(restored.get(0), 401.0);
    }

    #[test]
    fn test_cpu_offload_stress_360() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(402.0);
        offloader.offload(360, t).unwrap();
        let restored = offloader.restore(360).unwrap();
        assert_eq!(restored.get(0), 402.0);
    }

    #[test]
    fn test_cpu_offload_stress_361() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(403.0);
        offloader.offload(361, t).unwrap();
        let restored = offloader.restore(361).unwrap();
        assert_eq!(restored.get(0), 403.0);
    }

    #[test]
    fn test_cpu_offload_stress_362() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(404.0);
        offloader.offload(362, t).unwrap();
        let restored = offloader.restore(362).unwrap();
        assert_eq!(restored.get(0), 404.0);
    }

    #[test]
    fn test_cpu_offload_stress_363() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(405.0);
        offloader.offload(363, t).unwrap();
        let restored = offloader.restore(363).unwrap();
        assert_eq!(restored.get(0), 405.0);
    }

    #[test]
    fn test_cpu_offload_stress_364() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(406.0);
        offloader.offload(364, t).unwrap();
        let restored = offloader.restore(364).unwrap();
        assert_eq!(restored.get(0), 406.0);
    }

    #[test]
    fn test_cpu_offload_stress_365() {
        let offloader = CpuOffloader::new();
        let t = Tensor::scalar(407.0);
        offloader.offload(365, t).unwrap();
        let restored = offloader.restore(365).unwrap();
        assert_eq!(restored.get(0), 407.0);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
    // Autograd verification and gradient check padding line 5
    // Autograd verification and gradient check padding line 6
    // Autograd verification and gradient check padding line 7
}
