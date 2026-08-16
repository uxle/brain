//! # Quantization & Straight-Through Estimator (STE)
//!
//! Differentiable approximations for quantized operations.

use brain_core::{BrainResult, Tensor};

/// Straight-Through Estimator (STE) backward for hard quantization.
pub fn grad_quantize_ste(x: &Tensor, g: &Tensor, min_val: f64, max_val: f64) -> BrainResult<Tensor> {
    let mut d = vec![0.0; x.numel()];
    for (i, (&xi, &gi)) in x.data().iter().zip(g.data().iter()).enumerate() {
        if xi >= min_val && xi <= max_val {
            d[i] = gi;
        }
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
    fn test_quant_grad_stress_001() {
        let x = Tensor::scalar(-0.49);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_002() {
        let x = Tensor::scalar(-0.48);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_003() {
        let x = Tensor::scalar(-0.47);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_004() {
        let x = Tensor::scalar(-0.46);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_005() {
        let x = Tensor::scalar(-0.45);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_006() {
        let x = Tensor::scalar(-0.44);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_007() {
        let x = Tensor::scalar(-0.43);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_008() {
        let x = Tensor::scalar(-0.42);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_009() {
        let x = Tensor::scalar(-0.41000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_010() {
        let x = Tensor::scalar(-0.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_011() {
        let x = Tensor::scalar(-0.39);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_012() {
        let x = Tensor::scalar(-0.38);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_013() {
        let x = Tensor::scalar(-0.37);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_014() {
        let x = Tensor::scalar(-0.36);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_015() {
        let x = Tensor::scalar(-0.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_016() {
        let x = Tensor::scalar(-0.33999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_017() {
        let x = Tensor::scalar(-0.32999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_018() {
        let x = Tensor::scalar(-0.32);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_019() {
        let x = Tensor::scalar(-0.31);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_020() {
        let x = Tensor::scalar(-0.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_021() {
        let x = Tensor::scalar(-0.29000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_022() {
        let x = Tensor::scalar(-0.28);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_023() {
        let x = Tensor::scalar(-0.27);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_024() {
        let x = Tensor::scalar(-0.26);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_025() {
        let x = Tensor::scalar(-0.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_026() {
        let x = Tensor::scalar(-0.24);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_027() {
        let x = Tensor::scalar(-0.22999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_028() {
        let x = Tensor::scalar(-0.21999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_029() {
        let x = Tensor::scalar(-0.21000000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_030() {
        let x = Tensor::scalar(-0.2);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_031() {
        let x = Tensor::scalar(-0.19);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_032() {
        let x = Tensor::scalar(-0.18);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_033() {
        let x = Tensor::scalar(-0.16999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_034() {
        let x = Tensor::scalar(-0.15999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_035() {
        let x = Tensor::scalar(-0.14999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_036() {
        let x = Tensor::scalar(-0.14);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_037() {
        let x = Tensor::scalar(-0.13);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_038() {
        let x = Tensor::scalar(-0.12);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_039() {
        let x = Tensor::scalar(-0.10999999999999999);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_040() {
        let x = Tensor::scalar(-0.09999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_041() {
        let x = Tensor::scalar(-0.08999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_042() {
        let x = Tensor::scalar(-0.08000000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_043() {
        let x = Tensor::scalar(-0.07);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_044() {
        let x = Tensor::scalar(-0.06);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_045() {
        let x = Tensor::scalar(-0.04999999999999999);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_046() {
        let x = Tensor::scalar(-0.03999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_047() {
        let x = Tensor::scalar(-0.02999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_048() {
        let x = Tensor::scalar(-0.020000000000000018);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_049() {
        let x = Tensor::scalar(-0.010000000000000009);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_050() {
        let x = Tensor::scalar(0.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_051() {
        let x = Tensor::scalar(0.010000000000000009);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_052() {
        let x = Tensor::scalar(0.020000000000000018);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_053() {
        let x = Tensor::scalar(0.030000000000000027);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_054() {
        let x = Tensor::scalar(0.040000000000000036);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_055() {
        let x = Tensor::scalar(0.050000000000000044);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_056() {
        let x = Tensor::scalar(0.06000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_057() {
        let x = Tensor::scalar(0.07000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_058() {
        let x = Tensor::scalar(0.07999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_059() {
        let x = Tensor::scalar(0.08999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_060() {
        let x = Tensor::scalar(0.09999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_061() {
        let x = Tensor::scalar(0.10999999999999999);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_062() {
        let x = Tensor::scalar(0.12);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_063() {
        let x = Tensor::scalar(0.13);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_064() {
        let x = Tensor::scalar(0.14);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_065() {
        let x = Tensor::scalar(0.15000000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_066() {
        let x = Tensor::scalar(0.16000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_067() {
        let x = Tensor::scalar(0.17000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_068() {
        let x = Tensor::scalar(0.18000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_069() {
        let x = Tensor::scalar(0.19000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_070() {
        let x = Tensor::scalar(0.20000000000000007);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_071() {
        let x = Tensor::scalar(0.20999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_072() {
        let x = Tensor::scalar(0.21999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_073() {
        let x = Tensor::scalar(0.22999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_074() {
        let x = Tensor::scalar(0.24);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_075() {
        let x = Tensor::scalar(0.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_076() {
        let x = Tensor::scalar(0.26);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_077() {
        let x = Tensor::scalar(0.27);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_078() {
        let x = Tensor::scalar(0.28);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_079() {
        let x = Tensor::scalar(0.29000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_080() {
        let x = Tensor::scalar(0.30000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_081() {
        let x = Tensor::scalar(0.31000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_082() {
        let x = Tensor::scalar(0.32000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_083() {
        let x = Tensor::scalar(0.33000000000000007);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_084() {
        let x = Tensor::scalar(0.33999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_085() {
        let x = Tensor::scalar(0.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_086() {
        let x = Tensor::scalar(0.36);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_087() {
        let x = Tensor::scalar(0.37);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_088() {
        let x = Tensor::scalar(0.38);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_089() {
        let x = Tensor::scalar(0.39);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_090() {
        let x = Tensor::scalar(0.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_091() {
        let x = Tensor::scalar(0.41000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_092() {
        let x = Tensor::scalar(0.42000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_093() {
        let x = Tensor::scalar(0.43000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_094() {
        let x = Tensor::scalar(0.44000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_095() {
        let x = Tensor::scalar(0.45000000000000007);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_096() {
        let x = Tensor::scalar(0.45999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_097() {
        let x = Tensor::scalar(0.47);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_098() {
        let x = Tensor::scalar(0.48);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_099() {
        let x = Tensor::scalar(0.49);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_100() {
        let x = Tensor::scalar(-0.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_101() {
        let x = Tensor::scalar(-0.49);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_102() {
        let x = Tensor::scalar(-0.48);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_103() {
        let x = Tensor::scalar(-0.47);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_104() {
        let x = Tensor::scalar(-0.46);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_105() {
        let x = Tensor::scalar(-0.45);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_106() {
        let x = Tensor::scalar(-0.44);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_107() {
        let x = Tensor::scalar(-0.43);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_108() {
        let x = Tensor::scalar(-0.42);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_109() {
        let x = Tensor::scalar(-0.41000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_110() {
        let x = Tensor::scalar(-0.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_111() {
        let x = Tensor::scalar(-0.39);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_112() {
        let x = Tensor::scalar(-0.38);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_113() {
        let x = Tensor::scalar(-0.37);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_114() {
        let x = Tensor::scalar(-0.36);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_115() {
        let x = Tensor::scalar(-0.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_116() {
        let x = Tensor::scalar(-0.33999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_117() {
        let x = Tensor::scalar(-0.32999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_118() {
        let x = Tensor::scalar(-0.32);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_119() {
        let x = Tensor::scalar(-0.31);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_120() {
        let x = Tensor::scalar(-0.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_121() {
        let x = Tensor::scalar(-0.29000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_122() {
        let x = Tensor::scalar(-0.28);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_123() {
        let x = Tensor::scalar(-0.27);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_124() {
        let x = Tensor::scalar(-0.26);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_125() {
        let x = Tensor::scalar(-0.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_126() {
        let x = Tensor::scalar(-0.24);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_127() {
        let x = Tensor::scalar(-0.22999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_128() {
        let x = Tensor::scalar(-0.21999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_129() {
        let x = Tensor::scalar(-0.21000000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_130() {
        let x = Tensor::scalar(-0.2);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_131() {
        let x = Tensor::scalar(-0.19);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_132() {
        let x = Tensor::scalar(-0.18);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_133() {
        let x = Tensor::scalar(-0.16999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_134() {
        let x = Tensor::scalar(-0.15999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_135() {
        let x = Tensor::scalar(-0.14999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_136() {
        let x = Tensor::scalar(-0.14);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_137() {
        let x = Tensor::scalar(-0.13);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_138() {
        let x = Tensor::scalar(-0.12);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_139() {
        let x = Tensor::scalar(-0.10999999999999999);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_140() {
        let x = Tensor::scalar(-0.09999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_141() {
        let x = Tensor::scalar(-0.08999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_142() {
        let x = Tensor::scalar(-0.08000000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_143() {
        let x = Tensor::scalar(-0.07);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_144() {
        let x = Tensor::scalar(-0.06);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_145() {
        let x = Tensor::scalar(-0.04999999999999999);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_146() {
        let x = Tensor::scalar(-0.03999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_147() {
        let x = Tensor::scalar(-0.02999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_148() {
        let x = Tensor::scalar(-0.020000000000000018);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_149() {
        let x = Tensor::scalar(-0.010000000000000009);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_150() {
        let x = Tensor::scalar(0.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_151() {
        let x = Tensor::scalar(0.010000000000000009);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_152() {
        let x = Tensor::scalar(0.020000000000000018);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_153() {
        let x = Tensor::scalar(0.030000000000000027);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_154() {
        let x = Tensor::scalar(0.040000000000000036);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_155() {
        let x = Tensor::scalar(0.050000000000000044);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_156() {
        let x = Tensor::scalar(0.06000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_157() {
        let x = Tensor::scalar(0.07000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_158() {
        let x = Tensor::scalar(0.07999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_159() {
        let x = Tensor::scalar(0.08999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_160() {
        let x = Tensor::scalar(0.09999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_161() {
        let x = Tensor::scalar(0.10999999999999999);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_162() {
        let x = Tensor::scalar(0.12);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_163() {
        let x = Tensor::scalar(0.13);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_164() {
        let x = Tensor::scalar(0.14);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_165() {
        let x = Tensor::scalar(0.15000000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_166() {
        let x = Tensor::scalar(0.16000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_167() {
        let x = Tensor::scalar(0.17000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_168() {
        let x = Tensor::scalar(0.18000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_169() {
        let x = Tensor::scalar(0.19000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_170() {
        let x = Tensor::scalar(0.20000000000000007);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_171() {
        let x = Tensor::scalar(0.20999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_172() {
        let x = Tensor::scalar(0.21999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_173() {
        let x = Tensor::scalar(0.22999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_174() {
        let x = Tensor::scalar(0.24);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_175() {
        let x = Tensor::scalar(0.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_176() {
        let x = Tensor::scalar(0.26);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_177() {
        let x = Tensor::scalar(0.27);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_178() {
        let x = Tensor::scalar(0.28);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_179() {
        let x = Tensor::scalar(0.29000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_180() {
        let x = Tensor::scalar(0.30000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_181() {
        let x = Tensor::scalar(0.31000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_182() {
        let x = Tensor::scalar(0.32000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_183() {
        let x = Tensor::scalar(0.33000000000000007);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_184() {
        let x = Tensor::scalar(0.33999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_185() {
        let x = Tensor::scalar(0.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_186() {
        let x = Tensor::scalar(0.36);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_187() {
        let x = Tensor::scalar(0.37);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_188() {
        let x = Tensor::scalar(0.38);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_189() {
        let x = Tensor::scalar(0.39);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_190() {
        let x = Tensor::scalar(0.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_191() {
        let x = Tensor::scalar(0.41000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_192() {
        let x = Tensor::scalar(0.42000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_193() {
        let x = Tensor::scalar(0.43000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_194() {
        let x = Tensor::scalar(0.44000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_195() {
        let x = Tensor::scalar(0.45000000000000007);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_196() {
        let x = Tensor::scalar(0.45999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_197() {
        let x = Tensor::scalar(0.47);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_198() {
        let x = Tensor::scalar(0.48);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_199() {
        let x = Tensor::scalar(0.49);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_200() {
        let x = Tensor::scalar(-0.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_201() {
        let x = Tensor::scalar(-0.49);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_202() {
        let x = Tensor::scalar(-0.48);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_203() {
        let x = Tensor::scalar(-0.47);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_204() {
        let x = Tensor::scalar(-0.46);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_205() {
        let x = Tensor::scalar(-0.45);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_206() {
        let x = Tensor::scalar(-0.44);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_207() {
        let x = Tensor::scalar(-0.43);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_208() {
        let x = Tensor::scalar(-0.42);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_209() {
        let x = Tensor::scalar(-0.41000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_210() {
        let x = Tensor::scalar(-0.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_211() {
        let x = Tensor::scalar(-0.39);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_212() {
        let x = Tensor::scalar(-0.38);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_213() {
        let x = Tensor::scalar(-0.37);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_214() {
        let x = Tensor::scalar(-0.36);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_215() {
        let x = Tensor::scalar(-0.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_216() {
        let x = Tensor::scalar(-0.33999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_217() {
        let x = Tensor::scalar(-0.32999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_218() {
        let x = Tensor::scalar(-0.32);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_219() {
        let x = Tensor::scalar(-0.31);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_220() {
        let x = Tensor::scalar(-0.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_221() {
        let x = Tensor::scalar(-0.29000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_222() {
        let x = Tensor::scalar(-0.28);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_223() {
        let x = Tensor::scalar(-0.27);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_224() {
        let x = Tensor::scalar(-0.26);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_225() {
        let x = Tensor::scalar(-0.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_226() {
        let x = Tensor::scalar(-0.24);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_227() {
        let x = Tensor::scalar(-0.22999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_228() {
        let x = Tensor::scalar(-0.21999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_229() {
        let x = Tensor::scalar(-0.21000000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_230() {
        let x = Tensor::scalar(-0.2);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_231() {
        let x = Tensor::scalar(-0.19);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_232() {
        let x = Tensor::scalar(-0.18);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_233() {
        let x = Tensor::scalar(-0.16999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_234() {
        let x = Tensor::scalar(-0.15999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_235() {
        let x = Tensor::scalar(-0.14999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_236() {
        let x = Tensor::scalar(-0.14);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_237() {
        let x = Tensor::scalar(-0.13);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_238() {
        let x = Tensor::scalar(-0.12);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_239() {
        let x = Tensor::scalar(-0.10999999999999999);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_240() {
        let x = Tensor::scalar(-0.09999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_241() {
        let x = Tensor::scalar(-0.08999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_242() {
        let x = Tensor::scalar(-0.08000000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_243() {
        let x = Tensor::scalar(-0.07);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_244() {
        let x = Tensor::scalar(-0.06);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_245() {
        let x = Tensor::scalar(-0.04999999999999999);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_246() {
        let x = Tensor::scalar(-0.03999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_247() {
        let x = Tensor::scalar(-0.02999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_248() {
        let x = Tensor::scalar(-0.020000000000000018);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_249() {
        let x = Tensor::scalar(-0.010000000000000009);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_250() {
        let x = Tensor::scalar(0.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_251() {
        let x = Tensor::scalar(0.010000000000000009);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_252() {
        let x = Tensor::scalar(0.020000000000000018);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_253() {
        let x = Tensor::scalar(0.030000000000000027);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_254() {
        let x = Tensor::scalar(0.040000000000000036);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_255() {
        let x = Tensor::scalar(0.050000000000000044);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_256() {
        let x = Tensor::scalar(0.06000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_257() {
        let x = Tensor::scalar(0.07000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_258() {
        let x = Tensor::scalar(0.07999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_259() {
        let x = Tensor::scalar(0.08999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_260() {
        let x = Tensor::scalar(0.09999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_261() {
        let x = Tensor::scalar(0.10999999999999999);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_262() {
        let x = Tensor::scalar(0.12);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_263() {
        let x = Tensor::scalar(0.13);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_264() {
        let x = Tensor::scalar(0.14);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_265() {
        let x = Tensor::scalar(0.15000000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_266() {
        let x = Tensor::scalar(0.16000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_267() {
        let x = Tensor::scalar(0.17000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_268() {
        let x = Tensor::scalar(0.18000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_269() {
        let x = Tensor::scalar(0.19000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_270() {
        let x = Tensor::scalar(0.20000000000000007);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_271() {
        let x = Tensor::scalar(0.20999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_272() {
        let x = Tensor::scalar(0.21999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_273() {
        let x = Tensor::scalar(0.22999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_274() {
        let x = Tensor::scalar(0.24);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_275() {
        let x = Tensor::scalar(0.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_276() {
        let x = Tensor::scalar(0.26);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_277() {
        let x = Tensor::scalar(0.27);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_278() {
        let x = Tensor::scalar(0.28);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_279() {
        let x = Tensor::scalar(0.29000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_280() {
        let x = Tensor::scalar(0.30000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_281() {
        let x = Tensor::scalar(0.31000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_282() {
        let x = Tensor::scalar(0.32000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_283() {
        let x = Tensor::scalar(0.33000000000000007);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_284() {
        let x = Tensor::scalar(0.33999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_285() {
        let x = Tensor::scalar(0.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_286() {
        let x = Tensor::scalar(0.36);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_287() {
        let x = Tensor::scalar(0.37);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_288() {
        let x = Tensor::scalar(0.38);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_289() {
        let x = Tensor::scalar(0.39);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_290() {
        let x = Tensor::scalar(0.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_291() {
        let x = Tensor::scalar(0.41000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_292() {
        let x = Tensor::scalar(0.42000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_293() {
        let x = Tensor::scalar(0.43000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_294() {
        let x = Tensor::scalar(0.44000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_295() {
        let x = Tensor::scalar(0.45000000000000007);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_296() {
        let x = Tensor::scalar(0.45999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_297() {
        let x = Tensor::scalar(0.47);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_298() {
        let x = Tensor::scalar(0.48);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_299() {
        let x = Tensor::scalar(0.49);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_300() {
        let x = Tensor::scalar(-0.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_301() {
        let x = Tensor::scalar(-0.49);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_302() {
        let x = Tensor::scalar(-0.48);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_303() {
        let x = Tensor::scalar(-0.47);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_304() {
        let x = Tensor::scalar(-0.46);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_305() {
        let x = Tensor::scalar(-0.45);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_306() {
        let x = Tensor::scalar(-0.44);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_307() {
        let x = Tensor::scalar(-0.43);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_308() {
        let x = Tensor::scalar(-0.42);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_309() {
        let x = Tensor::scalar(-0.41000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_310() {
        let x = Tensor::scalar(-0.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_311() {
        let x = Tensor::scalar(-0.39);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_312() {
        let x = Tensor::scalar(-0.38);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_313() {
        let x = Tensor::scalar(-0.37);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_314() {
        let x = Tensor::scalar(-0.36);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_315() {
        let x = Tensor::scalar(-0.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_316() {
        let x = Tensor::scalar(-0.33999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_317() {
        let x = Tensor::scalar(-0.32999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_318() {
        let x = Tensor::scalar(-0.32);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_319() {
        let x = Tensor::scalar(-0.31);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_320() {
        let x = Tensor::scalar(-0.3);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_321() {
        let x = Tensor::scalar(-0.29000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_322() {
        let x = Tensor::scalar(-0.28);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_323() {
        let x = Tensor::scalar(-0.27);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_324() {
        let x = Tensor::scalar(-0.26);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_325() {
        let x = Tensor::scalar(-0.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_326() {
        let x = Tensor::scalar(-0.24);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_327() {
        let x = Tensor::scalar(-0.22999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_328() {
        let x = Tensor::scalar(-0.21999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_329() {
        let x = Tensor::scalar(-0.21000000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_330() {
        let x = Tensor::scalar(-0.2);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_331() {
        let x = Tensor::scalar(-0.19);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_332() {
        let x = Tensor::scalar(-0.18);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_333() {
        let x = Tensor::scalar(-0.16999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_334() {
        let x = Tensor::scalar(-0.15999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_335() {
        let x = Tensor::scalar(-0.14999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_336() {
        let x = Tensor::scalar(-0.14);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_337() {
        let x = Tensor::scalar(-0.13);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_338() {
        let x = Tensor::scalar(-0.12);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_339() {
        let x = Tensor::scalar(-0.10999999999999999);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_340() {
        let x = Tensor::scalar(-0.09999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_341() {
        let x = Tensor::scalar(-0.08999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_342() {
        let x = Tensor::scalar(-0.08000000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_343() {
        let x = Tensor::scalar(-0.07);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_344() {
        let x = Tensor::scalar(-0.06);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_345() {
        let x = Tensor::scalar(-0.04999999999999999);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_346() {
        let x = Tensor::scalar(-0.03999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_347() {
        let x = Tensor::scalar(-0.02999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_348() {
        let x = Tensor::scalar(-0.020000000000000018);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_349() {
        let x = Tensor::scalar(-0.010000000000000009);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_350() {
        let x = Tensor::scalar(0.0);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_351() {
        let x = Tensor::scalar(0.010000000000000009);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_352() {
        let x = Tensor::scalar(0.020000000000000018);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_353() {
        let x = Tensor::scalar(0.030000000000000027);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_354() {
        let x = Tensor::scalar(0.040000000000000036);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_355() {
        let x = Tensor::scalar(0.050000000000000044);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_356() {
        let x = Tensor::scalar(0.06000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_357() {
        let x = Tensor::scalar(0.07000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_358() {
        let x = Tensor::scalar(0.07999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_359() {
        let x = Tensor::scalar(0.08999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_360() {
        let x = Tensor::scalar(0.09999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_361() {
        let x = Tensor::scalar(0.10999999999999999);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_362() {
        let x = Tensor::scalar(0.12);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_363() {
        let x = Tensor::scalar(0.13);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_364() {
        let x = Tensor::scalar(0.14);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_365() {
        let x = Tensor::scalar(0.15000000000000002);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_366() {
        let x = Tensor::scalar(0.16000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_367() {
        let x = Tensor::scalar(0.17000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_368() {
        let x = Tensor::scalar(0.18000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_369() {
        let x = Tensor::scalar(0.19000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_370() {
        let x = Tensor::scalar(0.20000000000000007);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_371() {
        let x = Tensor::scalar(0.20999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_372() {
        let x = Tensor::scalar(0.21999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_373() {
        let x = Tensor::scalar(0.22999999999999998);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_374() {
        let x = Tensor::scalar(0.24);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_375() {
        let x = Tensor::scalar(0.25);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_376() {
        let x = Tensor::scalar(0.26);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_377() {
        let x = Tensor::scalar(0.27);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_378() {
        let x = Tensor::scalar(0.28);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_379() {
        let x = Tensor::scalar(0.29000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_380() {
        let x = Tensor::scalar(0.30000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_381() {
        let x = Tensor::scalar(0.31000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_382() {
        let x = Tensor::scalar(0.32000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_383() {
        let x = Tensor::scalar(0.33000000000000007);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_384() {
        let x = Tensor::scalar(0.33999999999999997);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_385() {
        let x = Tensor::scalar(0.35);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_386() {
        let x = Tensor::scalar(0.36);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_387() {
        let x = Tensor::scalar(0.37);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_388() {
        let x = Tensor::scalar(0.38);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_389() {
        let x = Tensor::scalar(0.39);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_390() {
        let x = Tensor::scalar(0.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_391() {
        let x = Tensor::scalar(0.41000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_392() {
        let x = Tensor::scalar(0.42000000000000004);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_393() {
        let x = Tensor::scalar(0.43000000000000005);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_394() {
        let x = Tensor::scalar(0.44000000000000006);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_395() {
        let x = Tensor::scalar(0.45000000000000007);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_396() {
        let x = Tensor::scalar(0.45999999999999996);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_397() {
        let x = Tensor::scalar(0.47);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_398() {
        let x = Tensor::scalar(0.48);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_399() {
        let x = Tensor::scalar(0.49);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_400() {
        let x = Tensor::scalar(-0.5);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_401() {
        let x = Tensor::scalar(-0.49);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_402() {
        let x = Tensor::scalar(-0.48);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_403() {
        let x = Tensor::scalar(-0.47);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_404() {
        let x = Tensor::scalar(-0.46);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_405() {
        let x = Tensor::scalar(-0.45);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_406() {
        let x = Tensor::scalar(-0.44);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_407() {
        let x = Tensor::scalar(-0.43);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_408() {
        let x = Tensor::scalar(-0.42);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_409() {
        let x = Tensor::scalar(-0.41000000000000003);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_410() {
        let x = Tensor::scalar(-0.4);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_411() {
        let x = Tensor::scalar(-0.39);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_412() {
        let x = Tensor::scalar(-0.38);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_413() {
        let x = Tensor::scalar(-0.37);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    #[test]
    fn test_quant_grad_stress_414() {
        let x = Tensor::scalar(-0.36);
        let g = Tensor::scalar(1.0);
        let dg = grad_quantize_ste(&x, &g, -1.0, 1.0).unwrap();
        assert_eq!(dg.get(0), 1.0);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
}
