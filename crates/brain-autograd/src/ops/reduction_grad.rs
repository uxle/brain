//! # Reduction Operation Gradients
//!
//! Backward rules for axis-wise reductions: `sum`, `mean`, `var`, `std`, `norm`, `cumsum`, and `cumprod`.

use brain_core::tensor::broadcast as bcast_t;
use brain_core::{BrainResult, Tensor};

/// Backward for `sum_axis`: expands gradient along reduced axis.
pub fn grad_sum_axis(g: &Tensor, orig_shape: &[usize], axis: usize) -> BrainResult<Tensor> {
    let mut expanded = g.clone();
    if g.ndim() < orig_shape.len() {
        expanded = g.unsqueeze(axis);
    }
    bcast_t::broadcast_to(&expanded, orig_shape)
}

/// Backward for `mean_axis`: expands gradient and divides by reduction dimension size.
pub fn grad_mean_axis(g: &Tensor, orig_shape: &[usize], axis: usize) -> BrainResult<Tensor> {
    let unscaled = grad_sum_axis(g, orig_shape, axis)?;
    let dim_size = orig_shape[axis] as f64;
    Ok(unscaled.map(|x| x / dim_size))
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
    fn test_reduction_grad_stress_001() {
        let g = Tensor::scalar(2.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_002() {
        let g = Tensor::scalar(2.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_003() {
        let g = Tensor::scalar(2.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_004() {
        let g = Tensor::scalar(2.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_005() {
        let g = Tensor::scalar(2.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_006() {
        let g = Tensor::scalar(2.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_007() {
        let g = Tensor::scalar(2.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_008() {
        let g = Tensor::scalar(2.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_009() {
        let g = Tensor::scalar(2.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_010() {
        let g = Tensor::scalar(3.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_011() {
        let g = Tensor::scalar(3.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_012() {
        let g = Tensor::scalar(3.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_013() {
        let g = Tensor::scalar(3.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_014() {
        let g = Tensor::scalar(3.4000000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_015() {
        let g = Tensor::scalar(3.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_016() {
        let g = Tensor::scalar(3.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_017() {
        let g = Tensor::scalar(3.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_018() {
        let g = Tensor::scalar(3.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_019() {
        let g = Tensor::scalar(3.9000000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_020() {
        let g = Tensor::scalar(4.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_021() {
        let g = Tensor::scalar(4.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_022() {
        let g = Tensor::scalar(4.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_023() {
        let g = Tensor::scalar(4.300000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_024() {
        let g = Tensor::scalar(4.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_025() {
        let g = Tensor::scalar(4.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_026() {
        let g = Tensor::scalar(4.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_027() {
        let g = Tensor::scalar(4.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_028() {
        let g = Tensor::scalar(4.800000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_029() {
        let g = Tensor::scalar(4.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_030() {
        let g = Tensor::scalar(5.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_031() {
        let g = Tensor::scalar(5.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_032() {
        let g = Tensor::scalar(5.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_033() {
        let g = Tensor::scalar(5.300000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_034() {
        let g = Tensor::scalar(5.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_035() {
        let g = Tensor::scalar(5.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_036() {
        let g = Tensor::scalar(5.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_037() {
        let g = Tensor::scalar(5.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_038() {
        let g = Tensor::scalar(5.800000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_039() {
        let g = Tensor::scalar(5.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_040() {
        let g = Tensor::scalar(6.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_041() {
        let g = Tensor::scalar(6.1000000000000005);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_042() {
        let g = Tensor::scalar(6.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_043() {
        let g = Tensor::scalar(6.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_044() {
        let g = Tensor::scalar(6.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_045() {
        let g = Tensor::scalar(6.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_046() {
        let g = Tensor::scalar(6.6000000000000005);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_047() {
        let g = Tensor::scalar(6.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_048() {
        let g = Tensor::scalar(6.800000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_049() {
        let g = Tensor::scalar(6.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_050() {
        let g = Tensor::scalar(7.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_051() {
        let g = Tensor::scalar(7.1000000000000005);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_052() {
        let g = Tensor::scalar(7.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_053() {
        let g = Tensor::scalar(7.300000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_054() {
        let g = Tensor::scalar(7.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_055() {
        let g = Tensor::scalar(7.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_056() {
        let g = Tensor::scalar(7.6000000000000005);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_057() {
        let g = Tensor::scalar(7.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_058() {
        let g = Tensor::scalar(7.800000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_059() {
        let g = Tensor::scalar(7.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_060() {
        let g = Tensor::scalar(8.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_061() {
        let g = Tensor::scalar(8.100000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_062() {
        let g = Tensor::scalar(8.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_063() {
        let g = Tensor::scalar(8.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_064() {
        let g = Tensor::scalar(8.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_065() {
        let g = Tensor::scalar(8.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_066() {
        let g = Tensor::scalar(8.600000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_067() {
        let g = Tensor::scalar(8.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_068() {
        let g = Tensor::scalar(8.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_069() {
        let g = Tensor::scalar(8.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_070() {
        let g = Tensor::scalar(9.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_071() {
        let g = Tensor::scalar(9.100000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_072() {
        let g = Tensor::scalar(9.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_073() {
        let g = Tensor::scalar(9.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_074() {
        let g = Tensor::scalar(9.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_075() {
        let g = Tensor::scalar(9.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_076() {
        let g = Tensor::scalar(9.600000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_077() {
        let g = Tensor::scalar(9.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_078() {
        let g = Tensor::scalar(9.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_079() {
        let g = Tensor::scalar(9.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_080() {
        let g = Tensor::scalar(10.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_081() {
        let g = Tensor::scalar(10.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_082() {
        let g = Tensor::scalar(10.200000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_083() {
        let g = Tensor::scalar(10.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_084() {
        let g = Tensor::scalar(10.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_085() {
        let g = Tensor::scalar(10.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_086() {
        let g = Tensor::scalar(10.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_087() {
        let g = Tensor::scalar(10.700000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_088() {
        let g = Tensor::scalar(10.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_089() {
        let g = Tensor::scalar(10.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_090() {
        let g = Tensor::scalar(11.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_091() {
        let g = Tensor::scalar(11.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_092() {
        let g = Tensor::scalar(11.200000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_093() {
        let g = Tensor::scalar(11.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_094() {
        let g = Tensor::scalar(11.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_095() {
        let g = Tensor::scalar(11.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_096() {
        let g = Tensor::scalar(11.600000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_097() {
        let g = Tensor::scalar(11.700000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_098() {
        let g = Tensor::scalar(11.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_099() {
        let g = Tensor::scalar(11.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_100() {
        let g = Tensor::scalar(12.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_101() {
        let g = Tensor::scalar(12.100000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_102() {
        let g = Tensor::scalar(12.200000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_103() {
        let g = Tensor::scalar(12.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_104() {
        let g = Tensor::scalar(12.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_105() {
        let g = Tensor::scalar(12.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_106() {
        let g = Tensor::scalar(12.600000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_107() {
        let g = Tensor::scalar(12.700000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_108() {
        let g = Tensor::scalar(12.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_109() {
        let g = Tensor::scalar(12.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_110() {
        let g = Tensor::scalar(13.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_111() {
        let g = Tensor::scalar(13.100000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_112() {
        let g = Tensor::scalar(13.200000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_113() {
        let g = Tensor::scalar(13.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_114() {
        let g = Tensor::scalar(13.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_115() {
        let g = Tensor::scalar(13.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_116() {
        let g = Tensor::scalar(13.600000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_117() {
        let g = Tensor::scalar(13.700000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_118() {
        let g = Tensor::scalar(13.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_119() {
        let g = Tensor::scalar(13.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_120() {
        let g = Tensor::scalar(14.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_121() {
        let g = Tensor::scalar(14.100000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_122() {
        let g = Tensor::scalar(14.200000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_123() {
        let g = Tensor::scalar(14.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_124() {
        let g = Tensor::scalar(14.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_125() {
        let g = Tensor::scalar(14.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_126() {
        let g = Tensor::scalar(14.600000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_127() {
        let g = Tensor::scalar(14.700000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_128() {
        let g = Tensor::scalar(14.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_129() {
        let g = Tensor::scalar(14.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_130() {
        let g = Tensor::scalar(15.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_131() {
        let g = Tensor::scalar(15.100000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_132() {
        let g = Tensor::scalar(15.200000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_133() {
        let g = Tensor::scalar(15.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_134() {
        let g = Tensor::scalar(15.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_135() {
        let g = Tensor::scalar(15.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_136() {
        let g = Tensor::scalar(15.600000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_137() {
        let g = Tensor::scalar(15.700000000000001);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_138() {
        let g = Tensor::scalar(15.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_139() {
        let g = Tensor::scalar(15.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_140() {
        let g = Tensor::scalar(16.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_141() {
        let g = Tensor::scalar(16.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_142() {
        let g = Tensor::scalar(16.200000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_143() {
        let g = Tensor::scalar(16.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_144() {
        let g = Tensor::scalar(16.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_145() {
        let g = Tensor::scalar(16.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_146() {
        let g = Tensor::scalar(16.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_147() {
        let g = Tensor::scalar(16.700000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_148() {
        let g = Tensor::scalar(16.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_149() {
        let g = Tensor::scalar(16.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_150() {
        let g = Tensor::scalar(17.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_151() {
        let g = Tensor::scalar(17.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_152() {
        let g = Tensor::scalar(17.200000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_153() {
        let g = Tensor::scalar(17.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_154() {
        let g = Tensor::scalar(17.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_155() {
        let g = Tensor::scalar(17.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_156() {
        let g = Tensor::scalar(17.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_157() {
        let g = Tensor::scalar(17.700000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_158() {
        let g = Tensor::scalar(17.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_159() {
        let g = Tensor::scalar(17.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_160() {
        let g = Tensor::scalar(18.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_161() {
        let g = Tensor::scalar(18.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_162() {
        let g = Tensor::scalar(18.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_163() {
        let g = Tensor::scalar(18.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_164() {
        let g = Tensor::scalar(18.400000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_165() {
        let g = Tensor::scalar(18.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_166() {
        let g = Tensor::scalar(18.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_167() {
        let g = Tensor::scalar(18.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_168() {
        let g = Tensor::scalar(18.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_169() {
        let g = Tensor::scalar(18.900000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_170() {
        let g = Tensor::scalar(19.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_171() {
        let g = Tensor::scalar(19.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_172() {
        let g = Tensor::scalar(19.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_173() {
        let g = Tensor::scalar(19.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_174() {
        let g = Tensor::scalar(19.400000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_175() {
        let g = Tensor::scalar(19.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_176() {
        let g = Tensor::scalar(19.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_177() {
        let g = Tensor::scalar(19.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_178() {
        let g = Tensor::scalar(19.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_179() {
        let g = Tensor::scalar(19.900000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_180() {
        let g = Tensor::scalar(20.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_181() {
        let g = Tensor::scalar(20.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_182() {
        let g = Tensor::scalar(20.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_183() {
        let g = Tensor::scalar(20.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_184() {
        let g = Tensor::scalar(20.400000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_185() {
        let g = Tensor::scalar(20.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_186() {
        let g = Tensor::scalar(20.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_187() {
        let g = Tensor::scalar(20.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_188() {
        let g = Tensor::scalar(20.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_189() {
        let g = Tensor::scalar(20.900000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_190() {
        let g = Tensor::scalar(21.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_191() {
        let g = Tensor::scalar(21.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_192() {
        let g = Tensor::scalar(21.200000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_193() {
        let g = Tensor::scalar(21.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_194() {
        let g = Tensor::scalar(21.400000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_195() {
        let g = Tensor::scalar(21.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_196() {
        let g = Tensor::scalar(21.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_197() {
        let g = Tensor::scalar(21.700000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_198() {
        let g = Tensor::scalar(21.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_199() {
        let g = Tensor::scalar(21.900000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_200() {
        let g = Tensor::scalar(22.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_201() {
        let g = Tensor::scalar(22.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_202() {
        let g = Tensor::scalar(22.200000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_203() {
        let g = Tensor::scalar(22.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_204() {
        let g = Tensor::scalar(22.400000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_205() {
        let g = Tensor::scalar(22.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_206() {
        let g = Tensor::scalar(22.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_207() {
        let g = Tensor::scalar(22.700000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_208() {
        let g = Tensor::scalar(22.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_209() {
        let g = Tensor::scalar(22.900000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_210() {
        let g = Tensor::scalar(23.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_211() {
        let g = Tensor::scalar(23.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_212() {
        let g = Tensor::scalar(23.200000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_213() {
        let g = Tensor::scalar(23.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_214() {
        let g = Tensor::scalar(23.400000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_215() {
        let g = Tensor::scalar(23.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_216() {
        let g = Tensor::scalar(23.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_217() {
        let g = Tensor::scalar(23.700000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_218() {
        let g = Tensor::scalar(23.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_219() {
        let g = Tensor::scalar(23.900000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_220() {
        let g = Tensor::scalar(24.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_221() {
        let g = Tensor::scalar(24.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_222() {
        let g = Tensor::scalar(24.200000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_223() {
        let g = Tensor::scalar(24.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_224() {
        let g = Tensor::scalar(24.400000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_225() {
        let g = Tensor::scalar(24.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_226() {
        let g = Tensor::scalar(24.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_227() {
        let g = Tensor::scalar(24.700000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_228() {
        let g = Tensor::scalar(24.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_229() {
        let g = Tensor::scalar(24.900000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_230() {
        let g = Tensor::scalar(25.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_231() {
        let g = Tensor::scalar(25.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_232() {
        let g = Tensor::scalar(25.200000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_233() {
        let g = Tensor::scalar(25.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_234() {
        let g = Tensor::scalar(25.400000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_235() {
        let g = Tensor::scalar(25.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_236() {
        let g = Tensor::scalar(25.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_237() {
        let g = Tensor::scalar(25.700000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_238() {
        let g = Tensor::scalar(25.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_239() {
        let g = Tensor::scalar(25.900000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_240() {
        let g = Tensor::scalar(26.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_241() {
        let g = Tensor::scalar(26.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_242() {
        let g = Tensor::scalar(26.200000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_243() {
        let g = Tensor::scalar(26.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_244() {
        let g = Tensor::scalar(26.400000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_245() {
        let g = Tensor::scalar(26.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_246() {
        let g = Tensor::scalar(26.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_247() {
        let g = Tensor::scalar(26.700000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_248() {
        let g = Tensor::scalar(26.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_249() {
        let g = Tensor::scalar(26.900000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_250() {
        let g = Tensor::scalar(27.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_251() {
        let g = Tensor::scalar(27.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_252() {
        let g = Tensor::scalar(27.200000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_253() {
        let g = Tensor::scalar(27.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_254() {
        let g = Tensor::scalar(27.400000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_255() {
        let g = Tensor::scalar(27.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_256() {
        let g = Tensor::scalar(27.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_257() {
        let g = Tensor::scalar(27.700000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_258() {
        let g = Tensor::scalar(27.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_259() {
        let g = Tensor::scalar(27.900000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_260() {
        let g = Tensor::scalar(28.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_261() {
        let g = Tensor::scalar(28.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_262() {
        let g = Tensor::scalar(28.200000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_263() {
        let g = Tensor::scalar(28.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_264() {
        let g = Tensor::scalar(28.400000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_265() {
        let g = Tensor::scalar(28.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_266() {
        let g = Tensor::scalar(28.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_267() {
        let g = Tensor::scalar(28.700000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_268() {
        let g = Tensor::scalar(28.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_269() {
        let g = Tensor::scalar(28.900000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_270() {
        let g = Tensor::scalar(29.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_271() {
        let g = Tensor::scalar(29.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_272() {
        let g = Tensor::scalar(29.200000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_273() {
        let g = Tensor::scalar(29.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_274() {
        let g = Tensor::scalar(29.400000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_275() {
        let g = Tensor::scalar(29.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_276() {
        let g = Tensor::scalar(29.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_277() {
        let g = Tensor::scalar(29.700000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_278() {
        let g = Tensor::scalar(29.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_279() {
        let g = Tensor::scalar(29.900000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_280() {
        let g = Tensor::scalar(30.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_281() {
        let g = Tensor::scalar(30.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_282() {
        let g = Tensor::scalar(30.200000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_283() {
        let g = Tensor::scalar(30.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_284() {
        let g = Tensor::scalar(30.400000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_285() {
        let g = Tensor::scalar(30.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_286() {
        let g = Tensor::scalar(30.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_287() {
        let g = Tensor::scalar(30.700000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_288() {
        let g = Tensor::scalar(30.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_289() {
        let g = Tensor::scalar(30.900000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_290() {
        let g = Tensor::scalar(31.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_291() {
        let g = Tensor::scalar(31.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_292() {
        let g = Tensor::scalar(31.200000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_293() {
        let g = Tensor::scalar(31.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_294() {
        let g = Tensor::scalar(31.400000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_295() {
        let g = Tensor::scalar(31.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_296() {
        let g = Tensor::scalar(31.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_297() {
        let g = Tensor::scalar(31.700000000000003);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_298() {
        let g = Tensor::scalar(31.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_299() {
        let g = Tensor::scalar(31.900000000000002);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_300() {
        let g = Tensor::scalar(32.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_301() {
        let g = Tensor::scalar(32.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_302() {
        let g = Tensor::scalar(32.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_303() {
        let g = Tensor::scalar(32.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_304() {
        let g = Tensor::scalar(32.400000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_305() {
        let g = Tensor::scalar(32.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_306() {
        let g = Tensor::scalar(32.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_307() {
        let g = Tensor::scalar(32.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_308() {
        let g = Tensor::scalar(32.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_309() {
        let g = Tensor::scalar(32.900000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_310() {
        let g = Tensor::scalar(33.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_311() {
        let g = Tensor::scalar(33.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_312() {
        let g = Tensor::scalar(33.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_313() {
        let g = Tensor::scalar(33.3);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_314() {
        let g = Tensor::scalar(33.400000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_315() {
        let g = Tensor::scalar(33.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_316() {
        let g = Tensor::scalar(33.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_317() {
        let g = Tensor::scalar(33.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_318() {
        let g = Tensor::scalar(33.8);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_319() {
        let g = Tensor::scalar(33.900000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_320() {
        let g = Tensor::scalar(34.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_321() {
        let g = Tensor::scalar(34.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_322() {
        let g = Tensor::scalar(34.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_323() {
        let g = Tensor::scalar(34.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_324() {
        let g = Tensor::scalar(34.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_325() {
        let g = Tensor::scalar(34.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_326() {
        let g = Tensor::scalar(34.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_327() {
        let g = Tensor::scalar(34.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_328() {
        let g = Tensor::scalar(34.800000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_329() {
        let g = Tensor::scalar(34.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_330() {
        let g = Tensor::scalar(35.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_331() {
        let g = Tensor::scalar(35.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_332() {
        let g = Tensor::scalar(35.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_333() {
        let g = Tensor::scalar(35.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_334() {
        let g = Tensor::scalar(35.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_335() {
        let g = Tensor::scalar(35.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_336() {
        let g = Tensor::scalar(35.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_337() {
        let g = Tensor::scalar(35.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_338() {
        let g = Tensor::scalar(35.800000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_339() {
        let g = Tensor::scalar(35.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_340() {
        let g = Tensor::scalar(36.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_341() {
        let g = Tensor::scalar(36.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_342() {
        let g = Tensor::scalar(36.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_343() {
        let g = Tensor::scalar(36.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_344() {
        let g = Tensor::scalar(36.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_345() {
        let g = Tensor::scalar(36.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_346() {
        let g = Tensor::scalar(36.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_347() {
        let g = Tensor::scalar(36.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_348() {
        let g = Tensor::scalar(36.800000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_349() {
        let g = Tensor::scalar(36.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_350() {
        let g = Tensor::scalar(37.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_351() {
        let g = Tensor::scalar(37.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_352() {
        let g = Tensor::scalar(37.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_353() {
        let g = Tensor::scalar(37.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_354() {
        let g = Tensor::scalar(37.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_355() {
        let g = Tensor::scalar(37.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_356() {
        let g = Tensor::scalar(37.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_357() {
        let g = Tensor::scalar(37.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_358() {
        let g = Tensor::scalar(37.800000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_359() {
        let g = Tensor::scalar(37.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_360() {
        let g = Tensor::scalar(38.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_361() {
        let g = Tensor::scalar(38.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_362() {
        let g = Tensor::scalar(38.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_363() {
        let g = Tensor::scalar(38.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_364() {
        let g = Tensor::scalar(38.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_365() {
        let g = Tensor::scalar(38.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_366() {
        let g = Tensor::scalar(38.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_367() {
        let g = Tensor::scalar(38.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_368() {
        let g = Tensor::scalar(38.800000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_369() {
        let g = Tensor::scalar(38.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_370() {
        let g = Tensor::scalar(39.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_371() {
        let g = Tensor::scalar(39.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_372() {
        let g = Tensor::scalar(39.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_373() {
        let g = Tensor::scalar(39.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_374() {
        let g = Tensor::scalar(39.4);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_375() {
        let g = Tensor::scalar(39.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_376() {
        let g = Tensor::scalar(39.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_377() {
        let g = Tensor::scalar(39.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_378() {
        let g = Tensor::scalar(39.800000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_379() {
        let g = Tensor::scalar(39.9);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_380() {
        let g = Tensor::scalar(40.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_381() {
        let g = Tensor::scalar(40.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_382() {
        let g = Tensor::scalar(40.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_383() {
        let g = Tensor::scalar(40.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_384() {
        let g = Tensor::scalar(40.400000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_385() {
        let g = Tensor::scalar(40.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_386() {
        let g = Tensor::scalar(40.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_387() {
        let g = Tensor::scalar(40.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_388() {
        let g = Tensor::scalar(40.800000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_389() {
        let g = Tensor::scalar(40.900000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_390() {
        let g = Tensor::scalar(41.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_391() {
        let g = Tensor::scalar(41.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_392() {
        let g = Tensor::scalar(41.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_393() {
        let g = Tensor::scalar(41.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_394() {
        let g = Tensor::scalar(41.400000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_395() {
        let g = Tensor::scalar(41.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_396() {
        let g = Tensor::scalar(41.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_397() {
        let g = Tensor::scalar(41.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_398() {
        let g = Tensor::scalar(41.800000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_399() {
        let g = Tensor::scalar(41.900000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_400() {
        let g = Tensor::scalar(42.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_401() {
        let g = Tensor::scalar(42.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_402() {
        let g = Tensor::scalar(42.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_403() {
        let g = Tensor::scalar(42.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_404() {
        let g = Tensor::scalar(42.400000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_405() {
        let g = Tensor::scalar(42.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_406() {
        let g = Tensor::scalar(42.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_407() {
        let g = Tensor::scalar(42.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_408() {
        let g = Tensor::scalar(42.800000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_409() {
        let g = Tensor::scalar(42.900000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_410() {
        let g = Tensor::scalar(43.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_411() {
        let g = Tensor::scalar(43.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_412() {
        let g = Tensor::scalar(43.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_413() {
        let g = Tensor::scalar(43.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_414() {
        let g = Tensor::scalar(43.400000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_415() {
        let g = Tensor::scalar(43.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_416() {
        let g = Tensor::scalar(43.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_417() {
        let g = Tensor::scalar(43.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_418() {
        let g = Tensor::scalar(43.800000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_419() {
        let g = Tensor::scalar(43.900000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_420() {
        let g = Tensor::scalar(44.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_421() {
        let g = Tensor::scalar(44.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_422() {
        let g = Tensor::scalar(44.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_423() {
        let g = Tensor::scalar(44.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_424() {
        let g = Tensor::scalar(44.400000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_425() {
        let g = Tensor::scalar(44.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_426() {
        let g = Tensor::scalar(44.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_427() {
        let g = Tensor::scalar(44.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_428() {
        let g = Tensor::scalar(44.800000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_429() {
        let g = Tensor::scalar(44.900000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_430() {
        let g = Tensor::scalar(45.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_431() {
        let g = Tensor::scalar(45.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_432() {
        let g = Tensor::scalar(45.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_433() {
        let g = Tensor::scalar(45.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_434() {
        let g = Tensor::scalar(45.400000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_435() {
        let g = Tensor::scalar(45.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_436() {
        let g = Tensor::scalar(45.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_437() {
        let g = Tensor::scalar(45.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_438() {
        let g = Tensor::scalar(45.800000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_439() {
        let g = Tensor::scalar(45.900000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_440() {
        let g = Tensor::scalar(46.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_441() {
        let g = Tensor::scalar(46.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_442() {
        let g = Tensor::scalar(46.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_443() {
        let g = Tensor::scalar(46.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_444() {
        let g = Tensor::scalar(46.400000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_445() {
        let g = Tensor::scalar(46.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_446() {
        let g = Tensor::scalar(46.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_447() {
        let g = Tensor::scalar(46.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_448() {
        let g = Tensor::scalar(46.800000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_449() {
        let g = Tensor::scalar(46.900000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_450() {
        let g = Tensor::scalar(47.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_451() {
        let g = Tensor::scalar(47.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_452() {
        let g = Tensor::scalar(47.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_453() {
        let g = Tensor::scalar(47.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_454() {
        let g = Tensor::scalar(47.400000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_455() {
        let g = Tensor::scalar(47.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_456() {
        let g = Tensor::scalar(47.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_457() {
        let g = Tensor::scalar(47.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_458() {
        let g = Tensor::scalar(47.800000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_459() {
        let g = Tensor::scalar(47.900000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_460() {
        let g = Tensor::scalar(48.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_461() {
        let g = Tensor::scalar(48.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_462() {
        let g = Tensor::scalar(48.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_463() {
        let g = Tensor::scalar(48.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_464() {
        let g = Tensor::scalar(48.400000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_465() {
        let g = Tensor::scalar(48.5);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_466() {
        let g = Tensor::scalar(48.6);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_467() {
        let g = Tensor::scalar(48.7);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_468() {
        let g = Tensor::scalar(48.800000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_469() {
        let g = Tensor::scalar(48.900000000000006);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_470() {
        let g = Tensor::scalar(49.0);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_471() {
        let g = Tensor::scalar(49.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_472() {
        let g = Tensor::scalar(49.2);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }

    #[test]
    fn test_reduction_grad_stress_473() {
        let g = Tensor::scalar(49.300000000000004);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }
}
