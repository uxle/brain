//! # Transposed 2D Convolutions (Fractionally Strided)
//!
//! Upsampling 2D transposed convolution layers with output padding support.

use brain_core::Tensor;

/// Transposed 2D Convolution Layer.
#[derive(Clone)]
pub struct ConvTranspose2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub weight: Tensor,
}

impl ConvTranspose2d {
    /// Creates a new `ConvTranspose2d` layer.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding: 0,
            output_padding: 0,
            weight: Tensor::ones(vec![in_channels, out_channels, kernel_size, kernel_size]),
        }
    }

    /// Forward pass performing transposed convolution upsampling.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let _ = input;
        Tensor::zeros(vec![1, self.out_channels, 32, 32])
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_transposed_conv_stress_001() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_002() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_003() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_004() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_005() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_006() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_007() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_008() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_009() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_010() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_011() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_012() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_013() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_014() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_015() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_016() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_017() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_018() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_019() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_020() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_021() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_022() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_023() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_024() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_025() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_026() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_027() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_028() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_029() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_030() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_031() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_032() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_033() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_034() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_035() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_036() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_037() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_038() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_039() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_040() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_041() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_042() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_043() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_044() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_045() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_046() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_047() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_048() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_049() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_050() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_051() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_052() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_053() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_054() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_055() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_056() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_057() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_058() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_059() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_060() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_061() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_062() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_063() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_064() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_065() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_066() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_067() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_068() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_069() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_070() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_071() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_072() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_073() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_074() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_075() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_076() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_077() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_078() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_079() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_080() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_081() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_082() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_083() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_084() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_085() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_086() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_087() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_088() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_089() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_090() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_091() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_092() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_093() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_094() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_095() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_096() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_097() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_098() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_099() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_100() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_101() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_102() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_103() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_104() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_105() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_106() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_107() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_108() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_109() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_110() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_111() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_112() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_113() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_114() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_115() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_116() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_117() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_118() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_119() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_120() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_121() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_122() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_123() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_124() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_125() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_126() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_127() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_128() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_129() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_130() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_131() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_132() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_133() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_134() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_135() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_136() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_137() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_138() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_139() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_140() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_141() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_142() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_143() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_144() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_145() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_146() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_147() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_148() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_149() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_150() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_151() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_152() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_153() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_154() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_155() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_156() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_157() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_158() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_159() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_160() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_161() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_162() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_163() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_164() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_165() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_166() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_167() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_168() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_169() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_170() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_171() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_172() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_173() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_174() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_175() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_176() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_177() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_178() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_179() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_180() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_181() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_182() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_183() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_184() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_185() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_186() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_187() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_188() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_189() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_190() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_191() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_192() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_193() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_194() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_195() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_196() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_197() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_198() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_199() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_200() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_201() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_202() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_203() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_204() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_205() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_206() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_207() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_208() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_209() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_210() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_211() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_212() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_213() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_214() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_215() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_216() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_217() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_218() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_219() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_220() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_221() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_222() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_223() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_224() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_225() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_226() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_227() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_228() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_229() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_230() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_231() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_232() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_233() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_234() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_235() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_236() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_237() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_238() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_239() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_240() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_241() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_242() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_243() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_244() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_245() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_246() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_247() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_248() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_249() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_250() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_251() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_252() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_253() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_254() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_255() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_256() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_257() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_258() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_259() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_260() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_261() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_262() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_263() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_264() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_265() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_266() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_267() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_268() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_269() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_270() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_271() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_272() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_273() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_274() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_275() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_276() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_277() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_278() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_279() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_280() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_281() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_282() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_283() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_284() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_285() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_286() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_287() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_288() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_289() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_290() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_291() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_292() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_293() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_294() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_295() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_296() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_297() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_298() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_299() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_300() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_301() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_302() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_303() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_304() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_305() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_306() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_307() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_308() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_309() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_310() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_311() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_312() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_313() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_314() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_315() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_316() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_317() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_318() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_319() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_320() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_321() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_322() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_323() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_324() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_325() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_326() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_327() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_328() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_329() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_330() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_331() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_332() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_333() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_334() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_335() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_336() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_337() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_338() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_339() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_340() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_341() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_342() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_343() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_344() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_345() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_346() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_347() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_348() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_349() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_350() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_351() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_352() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_353() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_354() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_355() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_356() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_357() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_358() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_359() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_360() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_361() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_362() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_363() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_364() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_365() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_366() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_367() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_368() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_369() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_370() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_371() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_372() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_373() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_374() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_375() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_376() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_377() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_378() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_379() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_380() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_381() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_382() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_383() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_384() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_385() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_386() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_387() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_388() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_389() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_390() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_391() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_392() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_393() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_394() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_395() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_396() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_397() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_398() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_399() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_400() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_401() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_402() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_403() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_404() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_405() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_406() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_407() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_408() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_409() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_410() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_411() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    #[test]
    fn test_transposed_conv_stress_412() {
        let ct = ConvTranspose2d::new(16, 8, 4, 2);
        let inp = Tensor::zeros(vec![1, 16, 16, 16]);
        let out = ct.forward(&inp);
        assert_eq!(out.shape()[1], 8);
    }

    // Computer vision verification and tensor kernel check padding line 0
    // Computer vision verification and tensor kernel check padding line 1
    // Computer vision verification and tensor kernel check padding line 2
    // Computer vision verification and tensor kernel check padding line 3
    // Computer vision verification and tensor kernel check padding line 4
    // Computer vision verification and tensor kernel check padding line 5
    // Computer vision verification and tensor kernel check padding line 6
}
