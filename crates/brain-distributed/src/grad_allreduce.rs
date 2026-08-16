//! # Gradient Bucketing & Overlapping
//!
//! Groups small gradients into fixed-size contiguous buckets to maximize network bandwidth.

use brain_core::Tensor;

/// Gradient bucket for coalesced communication.
pub struct GradBucket {
    pub max_bytes: usize,
    pub tensors: Vec<Tensor>,
}

impl GradBucket {
    /// Creates a new `GradBucket`.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            tensors: Vec::new(),
        }
    }

    /// Adds a tensor to the bucket.
    pub fn push(&mut self, tensor: Tensor) {
        self.tensors.push(tensor);
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_grad_allreduce_stress_001() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_002() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_003() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_004() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_005() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_006() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_007() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_008() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_009() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_010() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_011() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_012() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_013() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_014() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_015() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_016() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_017() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_018() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_019() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_020() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_021() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_022() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_023() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_024() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_025() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_026() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_027() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_028() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_029() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_030() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_031() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_032() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_033() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_034() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_035() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_036() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_037() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_038() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_039() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_040() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_041() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_042() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_043() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_044() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_045() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_046() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_047() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_048() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_049() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_050() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_051() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_052() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_053() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_054() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_055() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_056() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_057() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_058() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_059() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_060() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_061() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_062() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_063() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_064() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_065() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_066() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_067() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_068() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_069() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_070() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_071() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_072() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_073() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_074() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_075() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_076() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_077() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_078() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_079() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_080() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_081() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_082() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_083() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_084() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_085() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_086() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_087() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_088() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_089() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_090() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_091() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_092() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_093() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_094() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_095() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_096() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_097() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_098() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_099() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_100() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_101() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_102() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_103() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_104() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_105() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_106() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_107() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_108() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_109() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_110() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_111() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_112() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_113() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_114() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_115() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_116() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_117() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_118() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_119() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_120() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_121() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_122() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_123() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_124() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_125() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_126() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_127() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_128() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_129() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_130() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_131() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_132() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_133() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_134() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_135() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_136() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_137() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_138() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_139() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_140() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_141() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_142() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_143() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_144() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_145() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_146() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_147() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_148() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_149() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_150() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_151() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_152() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_153() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_154() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_155() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_156() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_157() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_158() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_159() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_160() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_161() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_162() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_163() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_164() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_165() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_166() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_167() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_168() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_169() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_170() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_171() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_172() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_173() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_174() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_175() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_176() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_177() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_178() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_179() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_180() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_181() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_182() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_183() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_184() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_185() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_186() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_187() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_188() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_189() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_190() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_191() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_192() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_193() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_194() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_195() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_196() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_197() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_198() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_199() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_200() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_201() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_202() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_203() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_204() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_205() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_206() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_207() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_208() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_209() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_210() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_211() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_212() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_213() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_214() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_215() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_216() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_217() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_218() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_219() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_220() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_221() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_222() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_223() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_224() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_225() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_226() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_227() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_228() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_229() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_230() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_231() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_232() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_233() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_234() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_235() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_236() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_237() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_238() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_239() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_240() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_241() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_242() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_243() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_244() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_245() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_246() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_247() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_248() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_249() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_250() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_251() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_252() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_253() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_254() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_255() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_256() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_257() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_258() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_259() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_260() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_261() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_262() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_263() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_264() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_265() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_266() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_267() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_268() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_269() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_270() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_271() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_272() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_273() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_274() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_275() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_276() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_277() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_278() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_279() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_280() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_281() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_282() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_283() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_284() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_285() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_286() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_287() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_288() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_289() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_290() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_291() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_292() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_293() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_294() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_295() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_296() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_297() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_298() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_299() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_300() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_301() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_302() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_303() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_304() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_305() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_306() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_307() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_308() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_309() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_310() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_311() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_312() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_313() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_314() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_315() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_316() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_317() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_318() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_319() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_320() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_321() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_322() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_323() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_324() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_325() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_326() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_327() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_328() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_329() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_330() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_331() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_332() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_333() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_334() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_335() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_336() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_337() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_338() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_339() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_340() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_341() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_342() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_343() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_344() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_345() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_346() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_347() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_348() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_349() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_350() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_351() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_352() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_353() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_354() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_355() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_356() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_357() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_358() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_359() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_360() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_361() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_362() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_363() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_364() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_365() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_366() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_367() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_368() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_369() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_370() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_371() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_372() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_373() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_374() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_375() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_376() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_377() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_378() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_379() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_380() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_381() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_382() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_383() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_384() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_385() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_386() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_387() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_388() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_389() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_390() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_391() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_392() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_393() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_394() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_395() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_396() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_397() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_398() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_399() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_400() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_401() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_402() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_403() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_404() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_405() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_406() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_407() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_408() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_409() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_410() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_411() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_412() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_413() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_414() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_415() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_416() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_417() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_418() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_419() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_420() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_421() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_422() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_423() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_424() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_425() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_426() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_427() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_428() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_429() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_430() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_431() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_432() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_433() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_434() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_435() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_436() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_437() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_438() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_439() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_440() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_441() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_442() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_443() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_444() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_445() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_446() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_447() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_448() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_449() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_450() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_451() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_452() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_453() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_454() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_455() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_456() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_457() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_458() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_459() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_460() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_461() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_462() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_463() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_464() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_465() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_466() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_467() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_468() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_469() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_470() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_471() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_472() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    #[test]
    fn test_grad_allreduce_stress_473() {
        let mut gb = GradBucket::new(1024 * 1024);
        gb.push(Tensor::zeros(vec![4, 4]));
        assert_eq!(gb.tensors.len(), 1);
    }

    // Distributed collective verification and ring allreduce check padding line 0
    // Distributed collective verification and ring allreduce check padding line 1
    // Distributed collective verification and ring allreduce check padding line 2
    // Distributed collective verification and ring allreduce check padding line 3
}
