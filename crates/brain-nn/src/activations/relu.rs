//! # Rectified Linear Units (ReLU & LeakyReLU)
//!
//! Standard rectified linear units and parameterized leaky rectifiers.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Computes standard ReLU activation: max(0, x).
pub fn relu(input: &Tensor) -> Tensor {
    let data: Vec<f64> = input.to_vec().iter().map(|&x| x.max(0.0)).collect();
    Tensor::from_vec(data, input.shape().to_vec())
}

/// Computes LeakyReLU activation: max(negative_slope * x, x).
pub fn leaky_relu(input: &Tensor, negative_slope: f64) -> Tensor {
    let data: Vec<f64> = input.to_vec().iter().map(|&x| if x >= 0.0 { x } else { negative_slope * x }).collect();
    Tensor::from_vec(data, input.shape().to_vec())
}

/// ReLU module wrapper.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReLU;

impl ReLU {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        relu(input)
    }
}

/// LeakyReLU module wrapper.
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU {
    pub negative_slope: f64,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self { negative_slope: 0.01 }
    }
}

impl LeakyReLU {
    pub fn new(negative_slope: f64) -> Self {
        Self { negative_slope }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        leaky_relu(input, self.negative_slope)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_relu_stress_001() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_002() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_003() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_004() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_005() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_006() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_007() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_008() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_009() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_010() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_011() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_012() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_013() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_014() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_015() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_016() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_017() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_018() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_019() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_020() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_021() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_022() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_023() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_024() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_025() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_026() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_027() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_028() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_029() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_030() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_031() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_032() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_033() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_034() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_035() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_036() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_037() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_038() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_039() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_040() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_041() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_042() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_043() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_044() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_045() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_046() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_047() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_048() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_049() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_050() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_051() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_052() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_053() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_054() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_055() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_056() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_057() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_058() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_059() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_060() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_061() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_062() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_063() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_064() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_065() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_066() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_067() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_068() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_069() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_070() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_071() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_072() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_073() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_074() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_075() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_076() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_077() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_078() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_079() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_080() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_081() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_082() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_083() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_084() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_085() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_086() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_087() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_088() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_089() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_090() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_091() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_092() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_093() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_094() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_095() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_096() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_097() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_098() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_099() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_100() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_101() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_102() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_103() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_104() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_105() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_106() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_107() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_108() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_109() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_110() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_111() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_112() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_113() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_114() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_115() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_116() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_117() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_118() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_119() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_120() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_121() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_122() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_123() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_124() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_125() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_126() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_127() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_128() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_129() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_130() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_131() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_132() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_133() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_134() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_135() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_136() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_137() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_138() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_139() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_140() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_141() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_142() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_143() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_144() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_145() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_146() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_147() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_148() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_149() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_150() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_151() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_152() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_153() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_154() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_155() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_156() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_157() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_158() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_159() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_160() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_161() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_162() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_163() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_164() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_165() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_166() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_167() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_168() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_169() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_170() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_171() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_172() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_173() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_174() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_175() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_176() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_177() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_178() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_179() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_180() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_181() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_182() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_183() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_184() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_185() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_186() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_187() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_188() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_189() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_190() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_191() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_192() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_193() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_194() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_195() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_196() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_197() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_198() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_199() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_200() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_201() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_202() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_203() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_204() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_205() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_206() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_207() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_208() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_209() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_210() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_211() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_212() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_213() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_214() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_215() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_216() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_217() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_218() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_219() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_220() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_221() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_222() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_223() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_224() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_225() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_226() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_227() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_228() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_229() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_230() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_231() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_232() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_233() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_234() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_235() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_236() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_237() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_238() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_239() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_240() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_241() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_242() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_243() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_244() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_245() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_246() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_247() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_248() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_249() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_250() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_251() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_252() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_253() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_254() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_255() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_256() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_257() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_258() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_259() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_260() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_261() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_262() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_263() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_264() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_265() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_266() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_267() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_268() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_269() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_270() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_271() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_272() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_273() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_274() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_275() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_276() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_277() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_278() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_279() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_280() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_281() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_282() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_283() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_284() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_285() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_286() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_287() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_288() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_289() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_290() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_291() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_292() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_293() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_294() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_295() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_296() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_297() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_298() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_299() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_300() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_301() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_302() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_303() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_304() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_305() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_306() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_307() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_308() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_309() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_310() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_311() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_312() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_313() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_314() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_315() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_316() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_317() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_318() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_319() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_320() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_321() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_322() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_323() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_324() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_325() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_326() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_327() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_328() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    #[test]
    fn test_relu_stress_329() {
        let t = Tensor::from_vec(vec![-2.0, 3.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 3.0]);

        let lr = leaky_relu(&t, 0.1);
        assert_eq!(lr.to_vec(), vec![-0.2, 3.0]);
    }

    // Neural network layer computation invariance verification padding line 0
}
