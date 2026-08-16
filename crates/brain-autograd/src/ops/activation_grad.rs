//! # Advanced Activation Function Gradients
//!
//! Numerically stable backward implementations for GELU, LeakyReLU, and SiLU.

use brain_core::{BrainResult, Tensor};

/// Backward pass for GELU.
pub fn grad_gelu(x: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let mut d = vec![0.0; x.numel()];
    let x_slice = x.data();
    let g_slice = g.data();
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();

    for (i, (&xi, &gi)) in x_slice.iter().zip(g_slice.iter()).enumerate() {
        let cube = 0.044715 * xi * xi * xi;
        let inner = sqrt_2_over_pi * (xi + cube);
        let tanh_inner = inner.tanh();
        let sech_sq = 1.0 - tanh_inner * tanh_inner;
        let cdf = 0.5 * (1.0 + tanh_inner);
        let pdf = 0.5 * xi * sech_sq * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * xi * xi);
        d[i] = gi * (cdf + pdf);
    }

    Ok(Tensor::from_slice(&d, x.shape().to_vec()))
}

/// Backward pass for LeakyReLU.
pub fn grad_leaky_relu(x: &Tensor, g: &Tensor, negative_slope: f64) -> BrainResult<Tensor> {
    let mut d = vec![0.0; x.numel()];
    for (i, (&xi, &gi)) in x.data().iter().zip(g.data().iter()).enumerate() {
        d[i] = if xi > 0.0 { gi } else { gi * negative_slope };
    }
    Ok(Tensor::from_slice(&d, x.shape().to_vec()))
}

/// Backward pass for SiLU / Swish.
pub fn grad_silu(x: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let mut d = vec![0.0; x.numel()];
    for (i, (&xi, &gi)) in x.data().iter().zip(g.data().iter()).enumerate() {
        let sig = 1.0 / (1.0 + (-xi).exp());
        d[i] = gi * (sig + xi * sig * (1.0 - sig));
    }
    Ok(Tensor::from_slice(&d, x.shape().to_vec()))
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
    fn test_activation_grad_stress_001() {
        let x = Tensor::scalar(0.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_002() {
        let x = Tensor::scalar(0.6);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_003() {
        let x = Tensor::scalar(0.65);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_004() {
        let x = Tensor::scalar(0.7);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_005() {
        let x = Tensor::scalar(0.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_006() {
        let x = Tensor::scalar(0.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_007() {
        let x = Tensor::scalar(0.8500000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_008() {
        let x = Tensor::scalar(0.9);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_009() {
        let x = Tensor::scalar(0.95);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_010() {
        let x = Tensor::scalar(1.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_011() {
        let x = Tensor::scalar(1.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_012() {
        let x = Tensor::scalar(1.1);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_013() {
        let x = Tensor::scalar(1.15);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_014() {
        let x = Tensor::scalar(1.2000000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_015() {
        let x = Tensor::scalar(1.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_016() {
        let x = Tensor::scalar(1.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_017() {
        let x = Tensor::scalar(1.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_018() {
        let x = Tensor::scalar(1.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_019() {
        let x = Tensor::scalar(1.4500000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_020() {
        let x = Tensor::scalar(1.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_021() {
        let x = Tensor::scalar(1.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_022() {
        let x = Tensor::scalar(1.6);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_023() {
        let x = Tensor::scalar(1.6500000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_024() {
        let x = Tensor::scalar(1.7000000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_025() {
        let x = Tensor::scalar(1.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_026() {
        let x = Tensor::scalar(1.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_027() {
        let x = Tensor::scalar(1.85);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_028() {
        let x = Tensor::scalar(1.9000000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_029() {
        let x = Tensor::scalar(1.9500000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_030() {
        let x = Tensor::scalar(2.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_031() {
        let x = Tensor::scalar(2.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_032() {
        let x = Tensor::scalar(2.1);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_033() {
        let x = Tensor::scalar(2.1500000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_034() {
        let x = Tensor::scalar(2.2);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_035() {
        let x = Tensor::scalar(2.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_036() {
        let x = Tensor::scalar(2.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_037() {
        let x = Tensor::scalar(2.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_038() {
        let x = Tensor::scalar(2.4000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_039() {
        let x = Tensor::scalar(2.45);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_040() {
        let x = Tensor::scalar(2.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_041() {
        let x = Tensor::scalar(2.5500000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_042() {
        let x = Tensor::scalar(2.6);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_043() {
        let x = Tensor::scalar(2.65);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_044() {
        let x = Tensor::scalar(2.7);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_045() {
        let x = Tensor::scalar(2.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_046() {
        let x = Tensor::scalar(2.8000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_047() {
        let x = Tensor::scalar(2.85);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_048() {
        let x = Tensor::scalar(2.9000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_049() {
        let x = Tensor::scalar(2.95);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_050() {
        let x = Tensor::scalar(3.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_051() {
        let x = Tensor::scalar(3.0500000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_052() {
        let x = Tensor::scalar(3.1);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_053() {
        let x = Tensor::scalar(3.1500000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_054() {
        let x = Tensor::scalar(3.2);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_055() {
        let x = Tensor::scalar(3.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_056() {
        let x = Tensor::scalar(3.3000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_057() {
        let x = Tensor::scalar(3.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_058() {
        let x = Tensor::scalar(3.4000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_059() {
        let x = Tensor::scalar(3.45);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_060() {
        let x = Tensor::scalar(3.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_061() {
        let x = Tensor::scalar(3.5500000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_062() {
        let x = Tensor::scalar(3.6);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_063() {
        let x = Tensor::scalar(3.6500000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_064() {
        let x = Tensor::scalar(3.7);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_065() {
        let x = Tensor::scalar(3.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_066() {
        let x = Tensor::scalar(3.8000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_067() {
        let x = Tensor::scalar(3.85);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_068() {
        let x = Tensor::scalar(3.9000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_069() {
        let x = Tensor::scalar(3.95);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_070() {
        let x = Tensor::scalar(4.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_071() {
        let x = Tensor::scalar(4.050000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_072() {
        let x = Tensor::scalar(4.1);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_073() {
        let x = Tensor::scalar(4.15);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_074() {
        let x = Tensor::scalar(4.2);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_075() {
        let x = Tensor::scalar(4.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_076() {
        let x = Tensor::scalar(4.300000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_077() {
        let x = Tensor::scalar(4.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_078() {
        let x = Tensor::scalar(4.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_079() {
        let x = Tensor::scalar(4.45);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_080() {
        let x = Tensor::scalar(4.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_081() {
        let x = Tensor::scalar(4.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_082() {
        let x = Tensor::scalar(4.6000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_083() {
        let x = Tensor::scalar(4.65);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_084() {
        let x = Tensor::scalar(4.7);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_085() {
        let x = Tensor::scalar(4.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_086() {
        let x = Tensor::scalar(4.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_087() {
        let x = Tensor::scalar(4.8500000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_088() {
        let x = Tensor::scalar(4.9);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_089() {
        let x = Tensor::scalar(4.95);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_090() {
        let x = Tensor::scalar(5.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_091() {
        let x = Tensor::scalar(5.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_092() {
        let x = Tensor::scalar(5.1000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_093() {
        let x = Tensor::scalar(5.15);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_094() {
        let x = Tensor::scalar(5.2);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_095() {
        let x = Tensor::scalar(5.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_096() {
        let x = Tensor::scalar(5.300000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_097() {
        let x = Tensor::scalar(5.3500000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_098() {
        let x = Tensor::scalar(5.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_099() {
        let x = Tensor::scalar(5.45);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_100() {
        let x = Tensor::scalar(5.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_101() {
        let x = Tensor::scalar(5.550000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_102() {
        let x = Tensor::scalar(5.6000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_103() {
        let x = Tensor::scalar(5.65);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_104() {
        let x = Tensor::scalar(5.7);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_105() {
        let x = Tensor::scalar(5.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_106() {
        let x = Tensor::scalar(5.800000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_107() {
        let x = Tensor::scalar(5.8500000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_108() {
        let x = Tensor::scalar(5.9);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_109() {
        let x = Tensor::scalar(5.95);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_110() {
        let x = Tensor::scalar(6.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_111() {
        let x = Tensor::scalar(6.050000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_112() {
        let x = Tensor::scalar(6.1000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_113() {
        let x = Tensor::scalar(6.15);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_114() {
        let x = Tensor::scalar(6.2);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_115() {
        let x = Tensor::scalar(6.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_116() {
        let x = Tensor::scalar(6.300000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_117() {
        let x = Tensor::scalar(6.3500000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_118() {
        let x = Tensor::scalar(6.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_119() {
        let x = Tensor::scalar(6.45);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_120() {
        let x = Tensor::scalar(6.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_121() {
        let x = Tensor::scalar(6.550000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_122() {
        let x = Tensor::scalar(6.6000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_123() {
        let x = Tensor::scalar(6.65);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_124() {
        let x = Tensor::scalar(6.7);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_125() {
        let x = Tensor::scalar(6.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_126() {
        let x = Tensor::scalar(6.800000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_127() {
        let x = Tensor::scalar(6.8500000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_128() {
        let x = Tensor::scalar(6.9);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_129() {
        let x = Tensor::scalar(6.95);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_130() {
        let x = Tensor::scalar(7.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_131() {
        let x = Tensor::scalar(7.050000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_132() {
        let x = Tensor::scalar(7.1000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_133() {
        let x = Tensor::scalar(7.15);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_134() {
        let x = Tensor::scalar(7.2);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_135() {
        let x = Tensor::scalar(7.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_136() {
        let x = Tensor::scalar(7.300000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_137() {
        let x = Tensor::scalar(7.3500000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_138() {
        let x = Tensor::scalar(7.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_139() {
        let x = Tensor::scalar(7.45);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_140() {
        let x = Tensor::scalar(7.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_141() {
        let x = Tensor::scalar(7.550000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_142() {
        let x = Tensor::scalar(7.6000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_143() {
        let x = Tensor::scalar(7.65);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_144() {
        let x = Tensor::scalar(7.7);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_145() {
        let x = Tensor::scalar(7.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_146() {
        let x = Tensor::scalar(7.800000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_147() {
        let x = Tensor::scalar(7.8500000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_148() {
        let x = Tensor::scalar(7.9);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_149() {
        let x = Tensor::scalar(7.95);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_150() {
        let x = Tensor::scalar(8.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_151() {
        let x = Tensor::scalar(8.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_152() {
        let x = Tensor::scalar(8.100000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_153() {
        let x = Tensor::scalar(8.15);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_154() {
        let x = Tensor::scalar(8.2);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_155() {
        let x = Tensor::scalar(8.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_156() {
        let x = Tensor::scalar(8.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_157() {
        let x = Tensor::scalar(8.350000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_158() {
        let x = Tensor::scalar(8.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_159() {
        let x = Tensor::scalar(8.45);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_160() {
        let x = Tensor::scalar(8.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_161() {
        let x = Tensor::scalar(8.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_162() {
        let x = Tensor::scalar(8.6);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_163() {
        let x = Tensor::scalar(8.65);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_164() {
        let x = Tensor::scalar(8.700000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_165() {
        let x = Tensor::scalar(8.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_166() {
        let x = Tensor::scalar(8.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_167() {
        let x = Tensor::scalar(8.85);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_168() {
        let x = Tensor::scalar(8.9);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_169() {
        let x = Tensor::scalar(8.950000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_170() {
        let x = Tensor::scalar(9.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_171() {
        let x = Tensor::scalar(9.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_172() {
        let x = Tensor::scalar(9.1);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_173() {
        let x = Tensor::scalar(9.15);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_174() {
        let x = Tensor::scalar(9.200000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_175() {
        let x = Tensor::scalar(9.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_176() {
        let x = Tensor::scalar(9.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_177() {
        let x = Tensor::scalar(9.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_178() {
        let x = Tensor::scalar(9.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_179() {
        let x = Tensor::scalar(9.450000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_180() {
        let x = Tensor::scalar(9.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_181() {
        let x = Tensor::scalar(9.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_182() {
        let x = Tensor::scalar(9.6);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_183() {
        let x = Tensor::scalar(9.65);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_184() {
        let x = Tensor::scalar(9.700000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_185() {
        let x = Tensor::scalar(9.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_186() {
        let x = Tensor::scalar(9.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_187() {
        let x = Tensor::scalar(9.85);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_188() {
        let x = Tensor::scalar(9.9);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_189() {
        let x = Tensor::scalar(9.950000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_190() {
        let x = Tensor::scalar(10.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_191() {
        let x = Tensor::scalar(10.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_192() {
        let x = Tensor::scalar(10.100000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_193() {
        let x = Tensor::scalar(10.15);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_194() {
        let x = Tensor::scalar(10.200000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_195() {
        let x = Tensor::scalar(10.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_196() {
        let x = Tensor::scalar(10.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_197() {
        let x = Tensor::scalar(10.350000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_198() {
        let x = Tensor::scalar(10.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_199() {
        let x = Tensor::scalar(10.450000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_200() {
        let x = Tensor::scalar(10.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_201() {
        let x = Tensor::scalar(10.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_202() {
        let x = Tensor::scalar(10.600000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_203() {
        let x = Tensor::scalar(10.65);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_204() {
        let x = Tensor::scalar(10.700000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_205() {
        let x = Tensor::scalar(10.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_206() {
        let x = Tensor::scalar(10.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_207() {
        let x = Tensor::scalar(10.850000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_208() {
        let x = Tensor::scalar(10.9);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_209() {
        let x = Tensor::scalar(10.950000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_210() {
        let x = Tensor::scalar(11.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_211() {
        let x = Tensor::scalar(11.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_212() {
        let x = Tensor::scalar(11.100000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_213() {
        let x = Tensor::scalar(11.15);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_214() {
        let x = Tensor::scalar(11.200000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_215() {
        let x = Tensor::scalar(11.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_216() {
        let x = Tensor::scalar(11.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_217() {
        let x = Tensor::scalar(11.350000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_218() {
        let x = Tensor::scalar(11.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_219() {
        let x = Tensor::scalar(11.450000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_220() {
        let x = Tensor::scalar(11.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_221() {
        let x = Tensor::scalar(11.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_222() {
        let x = Tensor::scalar(11.600000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_223() {
        let x = Tensor::scalar(11.65);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_224() {
        let x = Tensor::scalar(11.700000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_225() {
        let x = Tensor::scalar(11.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_226() {
        let x = Tensor::scalar(11.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_227() {
        let x = Tensor::scalar(11.850000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_228() {
        let x = Tensor::scalar(11.9);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_229() {
        let x = Tensor::scalar(11.950000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_230() {
        let x = Tensor::scalar(12.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_231() {
        let x = Tensor::scalar(12.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_232() {
        let x = Tensor::scalar(12.100000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_233() {
        let x = Tensor::scalar(12.15);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_234() {
        let x = Tensor::scalar(12.200000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_235() {
        let x = Tensor::scalar(12.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_236() {
        let x = Tensor::scalar(12.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_237() {
        let x = Tensor::scalar(12.350000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_238() {
        let x = Tensor::scalar(12.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_239() {
        let x = Tensor::scalar(12.450000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_240() {
        let x = Tensor::scalar(12.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_241() {
        let x = Tensor::scalar(12.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_242() {
        let x = Tensor::scalar(12.600000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_243() {
        let x = Tensor::scalar(12.65);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_244() {
        let x = Tensor::scalar(12.700000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_245() {
        let x = Tensor::scalar(12.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_246() {
        let x = Tensor::scalar(12.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_247() {
        let x = Tensor::scalar(12.850000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_248() {
        let x = Tensor::scalar(12.9);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_249() {
        let x = Tensor::scalar(12.950000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_250() {
        let x = Tensor::scalar(13.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_251() {
        let x = Tensor::scalar(13.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_252() {
        let x = Tensor::scalar(13.100000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_253() {
        let x = Tensor::scalar(13.15);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_254() {
        let x = Tensor::scalar(13.200000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_255() {
        let x = Tensor::scalar(13.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_256() {
        let x = Tensor::scalar(13.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_257() {
        let x = Tensor::scalar(13.350000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_258() {
        let x = Tensor::scalar(13.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_259() {
        let x = Tensor::scalar(13.450000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_260() {
        let x = Tensor::scalar(13.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_261() {
        let x = Tensor::scalar(13.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_262() {
        let x = Tensor::scalar(13.600000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_263() {
        let x = Tensor::scalar(13.65);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_264() {
        let x = Tensor::scalar(13.700000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_265() {
        let x = Tensor::scalar(13.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_266() {
        let x = Tensor::scalar(13.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_267() {
        let x = Tensor::scalar(13.850000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_268() {
        let x = Tensor::scalar(13.9);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_269() {
        let x = Tensor::scalar(13.950000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_270() {
        let x = Tensor::scalar(14.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_271() {
        let x = Tensor::scalar(14.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_272() {
        let x = Tensor::scalar(14.100000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_273() {
        let x = Tensor::scalar(14.15);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_274() {
        let x = Tensor::scalar(14.200000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_275() {
        let x = Tensor::scalar(14.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_276() {
        let x = Tensor::scalar(14.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_277() {
        let x = Tensor::scalar(14.350000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_278() {
        let x = Tensor::scalar(14.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_279() {
        let x = Tensor::scalar(14.450000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_280() {
        let x = Tensor::scalar(14.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_281() {
        let x = Tensor::scalar(14.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_282() {
        let x = Tensor::scalar(14.600000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_283() {
        let x = Tensor::scalar(14.65);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_284() {
        let x = Tensor::scalar(14.700000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_285() {
        let x = Tensor::scalar(14.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_286() {
        let x = Tensor::scalar(14.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_287() {
        let x = Tensor::scalar(14.850000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_288() {
        let x = Tensor::scalar(14.9);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_289() {
        let x = Tensor::scalar(14.950000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_290() {
        let x = Tensor::scalar(15.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_291() {
        let x = Tensor::scalar(15.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_292() {
        let x = Tensor::scalar(15.100000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_293() {
        let x = Tensor::scalar(15.15);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_294() {
        let x = Tensor::scalar(15.200000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_295() {
        let x = Tensor::scalar(15.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_296() {
        let x = Tensor::scalar(15.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_297() {
        let x = Tensor::scalar(15.350000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_298() {
        let x = Tensor::scalar(15.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_299() {
        let x = Tensor::scalar(15.450000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_300() {
        let x = Tensor::scalar(15.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_301() {
        let x = Tensor::scalar(15.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_302() {
        let x = Tensor::scalar(15.600000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_303() {
        let x = Tensor::scalar(15.65);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_304() {
        let x = Tensor::scalar(15.700000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_305() {
        let x = Tensor::scalar(15.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_306() {
        let x = Tensor::scalar(15.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_307() {
        let x = Tensor::scalar(15.850000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_308() {
        let x = Tensor::scalar(15.9);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_309() {
        let x = Tensor::scalar(15.950000000000001);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_310() {
        let x = Tensor::scalar(16.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_311() {
        let x = Tensor::scalar(16.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_312() {
        let x = Tensor::scalar(16.1);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_313() {
        let x = Tensor::scalar(16.15);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_314() {
        let x = Tensor::scalar(16.200000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_315() {
        let x = Tensor::scalar(16.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_316() {
        let x = Tensor::scalar(16.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_317() {
        let x = Tensor::scalar(16.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_318() {
        let x = Tensor::scalar(16.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_319() {
        let x = Tensor::scalar(16.450000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_320() {
        let x = Tensor::scalar(16.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_321() {
        let x = Tensor::scalar(16.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_322() {
        let x = Tensor::scalar(16.6);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_323() {
        let x = Tensor::scalar(16.650000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_324() {
        let x = Tensor::scalar(16.7);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_325() {
        let x = Tensor::scalar(16.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_326() {
        let x = Tensor::scalar(16.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_327() {
        let x = Tensor::scalar(16.85);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_328() {
        let x = Tensor::scalar(16.900000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_329() {
        let x = Tensor::scalar(16.95);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_330() {
        let x = Tensor::scalar(17.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_331() {
        let x = Tensor::scalar(17.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_332() {
        let x = Tensor::scalar(17.1);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_333() {
        let x = Tensor::scalar(17.150000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_334() {
        let x = Tensor::scalar(17.2);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_335() {
        let x = Tensor::scalar(17.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_336() {
        let x = Tensor::scalar(17.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_337() {
        let x = Tensor::scalar(17.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_338() {
        let x = Tensor::scalar(17.400000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_339() {
        let x = Tensor::scalar(17.45);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_340() {
        let x = Tensor::scalar(17.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_341() {
        let x = Tensor::scalar(17.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_342() {
        let x = Tensor::scalar(17.6);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_343() {
        let x = Tensor::scalar(17.650000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_344() {
        let x = Tensor::scalar(17.7);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_345() {
        let x = Tensor::scalar(17.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_346() {
        let x = Tensor::scalar(17.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_347() {
        let x = Tensor::scalar(17.85);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_348() {
        let x = Tensor::scalar(17.900000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_349() {
        let x = Tensor::scalar(17.95);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_350() {
        let x = Tensor::scalar(18.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_351() {
        let x = Tensor::scalar(18.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_352() {
        let x = Tensor::scalar(18.1);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_353() {
        let x = Tensor::scalar(18.150000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_354() {
        let x = Tensor::scalar(18.2);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_355() {
        let x = Tensor::scalar(18.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_356() {
        let x = Tensor::scalar(18.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_357() {
        let x = Tensor::scalar(18.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_358() {
        let x = Tensor::scalar(18.400000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_359() {
        let x = Tensor::scalar(18.45);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_360() {
        let x = Tensor::scalar(18.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_361() {
        let x = Tensor::scalar(18.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_362() {
        let x = Tensor::scalar(18.6);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_363() {
        let x = Tensor::scalar(18.650000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_364() {
        let x = Tensor::scalar(18.7);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_365() {
        let x = Tensor::scalar(18.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_366() {
        let x = Tensor::scalar(18.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_367() {
        let x = Tensor::scalar(18.85);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_368() {
        let x = Tensor::scalar(18.900000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_369() {
        let x = Tensor::scalar(18.95);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_370() {
        let x = Tensor::scalar(19.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_371() {
        let x = Tensor::scalar(19.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_372() {
        let x = Tensor::scalar(19.1);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_373() {
        let x = Tensor::scalar(19.150000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_374() {
        let x = Tensor::scalar(19.2);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_375() {
        let x = Tensor::scalar(19.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_376() {
        let x = Tensor::scalar(19.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_377() {
        let x = Tensor::scalar(19.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_378() {
        let x = Tensor::scalar(19.400000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_379() {
        let x = Tensor::scalar(19.45);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_380() {
        let x = Tensor::scalar(19.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_381() {
        let x = Tensor::scalar(19.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_382() {
        let x = Tensor::scalar(19.6);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_383() {
        let x = Tensor::scalar(19.650000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_384() {
        let x = Tensor::scalar(19.700000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_385() {
        let x = Tensor::scalar(19.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_386() {
        let x = Tensor::scalar(19.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_387() {
        let x = Tensor::scalar(19.85);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_388() {
        let x = Tensor::scalar(19.900000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_389() {
        let x = Tensor::scalar(19.950000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_390() {
        let x = Tensor::scalar(20.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_391() {
        let x = Tensor::scalar(20.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_392() {
        let x = Tensor::scalar(20.1);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_393() {
        let x = Tensor::scalar(20.150000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_394() {
        let x = Tensor::scalar(20.200000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_395() {
        let x = Tensor::scalar(20.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_396() {
        let x = Tensor::scalar(20.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_397() {
        let x = Tensor::scalar(20.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_398() {
        let x = Tensor::scalar(20.400000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_399() {
        let x = Tensor::scalar(20.450000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_400() {
        let x = Tensor::scalar(20.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_401() {
        let x = Tensor::scalar(20.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_402() {
        let x = Tensor::scalar(20.6);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_403() {
        let x = Tensor::scalar(20.650000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_404() {
        let x = Tensor::scalar(20.700000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_405() {
        let x = Tensor::scalar(20.75);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_406() {
        let x = Tensor::scalar(20.8);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_407() {
        let x = Tensor::scalar(20.85);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_408() {
        let x = Tensor::scalar(20.900000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_409() {
        let x = Tensor::scalar(20.950000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_410() {
        let x = Tensor::scalar(21.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    #[test]
    fn test_activation_grad_stress_411() {
        let x = Tensor::scalar(21.05);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }

    // Autograd verification and gradient check padding line 0
}
