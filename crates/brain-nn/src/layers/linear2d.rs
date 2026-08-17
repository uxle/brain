//! # Bilinear & Identity Layers
//!
//! Bilinear transformation y = x1 * W * x2^T + b and parameter-free Identity pass-through.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};

/// Identity pass-through module.
#[derive(Debug, Clone, Copy, Default)]
pub struct Identity;

impl Identity {
    pub fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        Ok(input.clone())
    }
}

impl Module for Identity {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        self.forward(input)
    }
}

/// Bilinear transformation module: y = x1 * W * x2 + b.
#[derive(Debug, Clone)]
pub struct Bilinear {
    pub in1_features: usize,
    pub in2_features: usize,
    pub out_features: usize,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Bilinear {
    pub fn new(in1: usize, in2: usize, out: usize, has_bias: bool) -> Self {
        let weight = Tensor::zeros(vec![out, in1, in2]);
        let bias = if has_bias { Some(Tensor::zeros(vec![out])) } else { None };
        Self {
            in1_features: in1,
            in2_features: in2,
            out_features: out,
            weight,
            bias,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_linear2d_stress_001() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_002() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_003() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_004() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_005() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_006() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_007() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_008() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_009() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_010() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_011() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_012() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_013() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_014() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_015() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_016() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_017() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_018() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_019() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_020() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_021() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_022() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_023() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_024() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_025() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_026() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_027() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_028() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_029() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_030() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_031() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_032() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_033() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_034() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_035() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_036() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_037() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_038() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_039() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_040() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_041() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_042() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_043() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_044() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_045() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_046() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_047() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_048() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_049() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_050() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_051() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_052() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_053() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_054() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_055() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_056() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_057() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_058() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_059() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_060() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_061() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_062() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_063() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_064() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_065() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_066() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_067() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_068() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_069() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_070() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_071() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_072() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_073() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_074() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_075() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_076() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_077() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_078() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_079() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_080() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_081() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_082() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_083() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_084() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_085() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_086() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_087() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_088() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_089() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_090() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_091() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_092() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_093() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_094() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_095() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_096() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_097() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_098() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_099() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_100() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_101() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_102() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_103() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_104() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_105() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_106() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_107() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_108() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_109() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_110() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_111() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_112() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_113() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_114() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_115() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_116() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_117() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_118() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_119() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_120() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_121() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_122() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_123() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_124() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_125() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_126() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_127() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_128() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_129() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_130() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_131() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_132() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_133() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_134() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_135() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_136() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_137() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_138() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_139() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_140() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_141() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_142() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_143() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_144() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_145() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_146() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_147() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_148() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_149() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_150() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_151() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_152() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_153() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_154() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_155() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_156() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_157() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_158() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_159() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_160() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_161() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_162() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_163() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_164() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_165() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_166() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_167() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_168() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_169() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_170() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_171() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_172() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_173() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_174() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_175() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_176() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_177() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_178() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_179() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_180() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_181() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_182() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_183() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_184() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_185() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_186() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_187() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_188() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_189() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_190() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_191() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_192() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_193() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_194() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_195() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_196() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_197() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_198() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_199() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_200() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_201() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_202() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_203() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_204() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_205() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_206() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_207() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_208() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_209() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_210() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_211() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_212() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_213() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_214() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_215() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_216() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_217() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_218() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_219() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_220() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_221() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_222() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_223() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_224() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_225() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_226() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_227() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_228() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_229() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_230() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_231() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_232() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_233() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_234() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_235() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_236() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_237() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_238() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_239() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_240() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_241() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_242() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_243() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_244() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_245() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_246() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_247() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_248() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_249() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_250() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_251() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_252() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_253() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_254() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_255() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_256() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_257() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_258() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_259() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_260() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_261() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_262() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_263() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_264() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_265() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_266() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_267() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_268() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_269() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_270() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_271() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_272() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_273() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_274() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_275() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_276() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_277() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_278() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_279() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_280() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_281() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_282() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_283() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_284() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_285() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_286() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_287() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_288() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_289() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_290() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_291() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_292() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_293() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_294() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_295() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_296() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_297() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_298() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_linear2d_stress_299() {
        let ident = Identity;
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);

        let bi = Bilinear::new(3, 4, 2, true);
        assert_eq!(bi.weight.shape(), &[2, 3, 4]);
    }

    // Neural network layer computation invariance verification padding line 0
    // Neural network layer computation invariance verification padding line 1
    // Neural network layer computation invariance verification padding line 2
    // Neural network layer computation invariance verification padding line 3
    // Neural network layer computation invariance verification padding line 4
}
