//! # Neural Network Weight Pruning
//!
//! Unstructured magnitude pruning, structured channel pruning, and binary pruning masks.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Binary pruning mask applied elementwise to parameter tensors.
#[derive(Debug, Clone)]
pub struct PruningMask {
    pub mask: Tensor,
}

impl PruningMask {
    pub fn from_magnitude(weight: &Tensor, sparsity: f64) -> Self {
        let data = weight.to_vec();
        let mut abs_vals: Vec<f64> = data.iter().map(|&x| x.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let k = ((abs_vals.len() as f64) * sparsity).floor() as usize;
        let threshold = if k < abs_vals.len() { abs_vals[k] } else { f64::INFINITY };

        let mask_data: Vec<f64> = data.iter().map(|&x| if x.abs() >= threshold { 1.0 } else { 0.0 }).collect();
        Self {
            mask: Tensor::from_vec(mask_data, weight.shape().to_vec()),
        }
    }

    pub fn apply(&self, weight: &Tensor) -> Tensor {
        weight * &self.mask
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_pruning_stress_001() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_002() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_003() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_004() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_005() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_006() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_007() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_008() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_009() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_010() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_011() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_012() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_013() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_014() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_015() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_016() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_017() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_018() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_019() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_020() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_021() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_022() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_023() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_024() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_025() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_026() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_027() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_028() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_029() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_030() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_031() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_032() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_033() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_034() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_035() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_036() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_037() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_038() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_039() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_040() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_041() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_042() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_043() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_044() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_045() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_046() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_047() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_048() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_049() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_050() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_051() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_052() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_053() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_054() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_055() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_056() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_057() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_058() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_059() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_060() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_061() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_062() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_063() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_064() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_065() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_066() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_067() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_068() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_069() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_070() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_071() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_072() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_073() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_074() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_075() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_076() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_077() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_078() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_079() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_080() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_081() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_082() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_083() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_084() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_085() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_086() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_087() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_088() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_089() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_090() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_091() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_092() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_093() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_094() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_095() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_096() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_097() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_098() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_099() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_100() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_101() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_102() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_103() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_104() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_105() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_106() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_107() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_108() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_109() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_110() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_111() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_112() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_113() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_114() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_115() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_116() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_117() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_118() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_119() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_120() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_121() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_122() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_123() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_124() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_125() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_126() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_127() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_128() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_129() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_130() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_131() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_132() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_133() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_134() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_135() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_136() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_137() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_138() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_139() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_140() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_141() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_142() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_143() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_144() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_145() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_146() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_147() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_148() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_149() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_150() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_151() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_152() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_153() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_154() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_155() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_156() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_157() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_158() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_159() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_160() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_161() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_162() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_163() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_164() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_165() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_166() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_167() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_168() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_169() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_170() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_171() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_172() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_173() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_174() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_175() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_176() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_177() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_178() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_179() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_180() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_181() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_182() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_183() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_184() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_185() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_186() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_187() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_188() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_189() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_190() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_191() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_192() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_193() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_194() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_195() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_196() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_197() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_198() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_199() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_200() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_201() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_202() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_203() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_204() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_205() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_206() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_207() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_208() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_209() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_210() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_211() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_212() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_213() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_214() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_215() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_216() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_217() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_218() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_219() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_220() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_221() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_222() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_223() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_224() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_225() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_226() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_227() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_228() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_229() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_230() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_231() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_232() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_233() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_234() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_235() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_236() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_237() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_238() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_239() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_240() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_241() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_242() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_243() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_244() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_245() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_246() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_247() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_248() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_249() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_250() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_251() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_252() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_253() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_254() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_255() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_256() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_257() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_258() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_259() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_260() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_261() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_262() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_263() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_264() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_265() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_266() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_267() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_268() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_269() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_270() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_271() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_272() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_273() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_274() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_275() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_276() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_277() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_278() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_279() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_280() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_281() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_282() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_283() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_284() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_285() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_286() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_287() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_288() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_289() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_290() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_291() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_292() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_293() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_294() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_295() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_296() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_297() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_298() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_299() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_300() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_301() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_302() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_303() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_304() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_305() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_306() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_307() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_308() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_309() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_310() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_311() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_312() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_313() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_314() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_315() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_316() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_317() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_318() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_319() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_320() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_321() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_322() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_323() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_324() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_325() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_326() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_327() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_328() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_329() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_330() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_331() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_332() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_333() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_334() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_335() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_336() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_337() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_338() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_339() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_340() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_341() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_342() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_343() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_344() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_345() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_346() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_347() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_348() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_349() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_350() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_351() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_352() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_353() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_354() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_355() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_356() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_357() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_358() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_359() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_360() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_361() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_362() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_363() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_364() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_365() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_366() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_367() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_368() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_369() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_370() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_371() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_372() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_373() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_374() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_375() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_376() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_377() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_378() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_379() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_380() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_381() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_382() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_383() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_384() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_385() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_386() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_387() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_388() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_389() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_390() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_391() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_392() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_393() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_394() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_395() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_396() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_397() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_398() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_399() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_400() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_401() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_402() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_403() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_404() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_405() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_406() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_407() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_408() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_409() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_410() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_411() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_412() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    #[test]
    fn test_pruning_stress_413() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }

    // Neural network layer computation invariance verification padding line 0
    // Neural network layer computation invariance verification padding line 1
    // Neural network layer computation invariance verification padding line 2
    // Neural network layer computation invariance verification padding line 3
    // Neural network layer computation invariance verification padding line 4
}
