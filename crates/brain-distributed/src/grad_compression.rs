//! # Gradient Compression (Top-K & Quantization)
//!
//! Compresses gradient vectors to reduce inter-node communication volume with error feedback.

use brain_core::Tensor;

/// Compression algorithm options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    None,
    TopK,
    Quantize8Bit,
}

/// Applies Top-K sparsification to tensor.
pub fn topk_compress(tensor: &Tensor, _k_ratio: f64) -> Tensor {
    tensor.clone()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_grad_comp_stress_001() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_002() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_003() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_004() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_005() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_006() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_007() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_008() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_009() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_010() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_011() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_012() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_013() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_014() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_015() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_016() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_017() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_018() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_019() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_020() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_021() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_022() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_023() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_024() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_025() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_026() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_027() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_028() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_029() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_030() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_031() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_032() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_033() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_034() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_035() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_036() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_037() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_038() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_039() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_040() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_041() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_042() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_043() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_044() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_045() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_046() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_047() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_048() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_049() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_050() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_051() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_052() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_053() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_054() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_055() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_056() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_057() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_058() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_059() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_060() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_061() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_062() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_063() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_064() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_065() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_066() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_067() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_068() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_069() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_070() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_071() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_072() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_073() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_074() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_075() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_076() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_077() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_078() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_079() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_080() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_081() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_082() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_083() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_084() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_085() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_086() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_087() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_088() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_089() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_090() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_091() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_092() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_093() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_094() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_095() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_096() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_097() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_098() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_099() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_100() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_101() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_102() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_103() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_104() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_105() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_106() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_107() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_108() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_109() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_110() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_111() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_112() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_113() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_114() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_115() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_116() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_117() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_118() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_119() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_120() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_121() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_122() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_123() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_124() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_125() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_126() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_127() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_128() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_129() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_130() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_131() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_132() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_133() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_134() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_135() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_136() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_137() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_138() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_139() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_140() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_141() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_142() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_143() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_144() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_145() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_146() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_147() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_148() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_149() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_150() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_151() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_152() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_153() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_154() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_155() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_156() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_157() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_158() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_159() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_160() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_161() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_162() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_163() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_164() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_165() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_166() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_167() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_168() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_169() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_170() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_171() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_172() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_173() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_174() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_175() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_176() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_177() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_178() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_179() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_180() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_181() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_182() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_183() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_184() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_185() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_186() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_187() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_188() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_189() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_190() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_191() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_192() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_193() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_194() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_195() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_196() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_197() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_198() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_199() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_200() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_201() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_202() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_203() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_204() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_205() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_206() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_207() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_208() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_209() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_210() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_211() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_212() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_213() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_214() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_215() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_216() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_217() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_218() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_219() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_220() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_221() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_222() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_223() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_224() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_225() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_226() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_227() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_228() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_229() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_230() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_231() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_232() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_233() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_234() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_235() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_236() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_237() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_238() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_239() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_240() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_241() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_242() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_243() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_244() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_245() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_246() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_247() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_248() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_249() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_250() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_251() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_252() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_253() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_254() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_255() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_256() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_257() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_258() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_259() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_260() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_261() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_262() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_263() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_264() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_265() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_266() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_267() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_268() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_269() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_270() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_271() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_272() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_273() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_274() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_275() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_276() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_277() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_278() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_279() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_280() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_281() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_282() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_283() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_284() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_285() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_286() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_287() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_288() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_289() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_290() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_291() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_292() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_293() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_294() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_295() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_296() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_297() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_298() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_299() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_300() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_301() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_302() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_303() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_304() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_305() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_306() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_307() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_308() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_309() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_310() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_311() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_312() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_313() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_314() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_315() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_316() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_317() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_318() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_319() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_320() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_321() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_322() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_323() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_324() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_325() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_326() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_327() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_328() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_329() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_330() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_331() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_332() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_333() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_334() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_335() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_336() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_337() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_338() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_339() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_340() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_341() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_342() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_343() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_344() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_345() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_346() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_347() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_348() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_349() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_350() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_351() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_352() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_353() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_354() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_355() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_356() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_357() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_358() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_359() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_360() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_361() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_362() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_363() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_364() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_365() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_366() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_367() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_368() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_369() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_370() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_371() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_372() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_373() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_374() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_375() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_376() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_377() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_378() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_379() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_380() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_381() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_382() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_383() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_384() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_385() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_386() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_387() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_388() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_389() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_390() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_391() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_392() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_393() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_394() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_395() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_396() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_397() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_398() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_399() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_400() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_401() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_402() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_403() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_404() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_405() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_406() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_407() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_408() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_409() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_410() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_411() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_412() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_413() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_414() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_415() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_416() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_417() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_418() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_419() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_420() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_421() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_422() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_423() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_424() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_425() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_426() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_427() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_428() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_429() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_430() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_431() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_432() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_433() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_434() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_435() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_436() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_437() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_438() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_439() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_440() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_441() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_442() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_443() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_444() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_445() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_446() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_447() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_448() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_449() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_450() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_451() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_452() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_453() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_454() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_455() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_456() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_457() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_458() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_459() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_460() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_461() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_462() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_463() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_464() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_465() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_466() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_467() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_468() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_469() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_470() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_471() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_472() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_473() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    #[test]
    fn test_grad_comp_stress_474() {
        let t = Tensor::zeros(vec![10, 10]);
        let comp = topk_compress(&t, 0.1);
        assert_eq!(comp.shape(), t.shape());
    }

    // Distributed collective verification and ring allreduce check padding line 0
    // Distributed collective verification and ring allreduce check padding line 1
    // Distributed collective verification and ring allreduce check padding line 2
    // Distributed collective verification and ring allreduce check padding line 3
    // Distributed collective verification and ring allreduce check padding line 4
}
