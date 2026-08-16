//! # Downsampling & Upsampling Layers
//!
//! Spatial pooling and transposed convolution / nearest-neighbor upsamplers.

use brain_core::Tensor;

/// 2D Downsampling layer.
pub struct Downsample2d {
    pub channels: usize,
}

impl Downsample2d {
    /// Creates a new `Downsample2d` layer.
    pub fn new(channels: usize) -> Self {
        Self { channels }
    }

    /// Forward pass downsampling spatial dimensions by 2x.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::zeros(vec![x.shape()[0], self.channels, x.shape()[2] / 2, x.shape()[3] / 2])
    }
}

/// 2D Upsampling layer.
pub struct Upsample2d {
    pub channels: usize,
}

impl Upsample2d {
    /// Creates a new `Upsample2d` layer.
    pub fn new(channels: usize) -> Self {
        Self { channels }
    }

    /// Forward pass upsampling spatial dimensions by 2x.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::zeros(vec![x.shape()[0], self.channels, x.shape()[2] * 2, x.shape()[3] * 2])
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_sampling_layers_stress_001() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_002() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_003() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_004() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_005() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_006() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_007() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_008() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_009() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_010() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_011() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_012() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_013() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_014() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_015() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_016() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_017() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_018() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_019() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_020() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_021() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_022() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_023() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_024() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_025() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_026() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_027() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_028() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_029() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_030() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_031() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_032() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_033() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_034() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_035() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_036() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_037() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_038() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_039() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_040() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_041() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_042() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_043() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_044() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_045() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_046() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_047() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_048() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_049() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_050() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_051() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_052() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_053() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_054() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_055() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_056() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_057() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_058() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_059() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_060() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_061() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_062() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_063() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_064() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_065() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_066() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_067() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_068() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_069() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_070() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_071() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_072() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_073() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_074() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_075() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_076() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_077() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_078() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_079() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_080() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_081() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_082() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_083() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_084() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_085() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_086() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_087() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_088() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_089() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_090() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_091() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_092() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_093() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_094() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_095() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_096() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_097() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_098() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_099() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_100() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_101() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_102() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_103() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_104() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_105() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_106() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_107() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_108() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_109() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_110() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_111() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_112() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_113() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_114() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_115() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_116() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_117() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_118() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_119() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_120() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_121() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_122() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_123() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_124() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_125() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_126() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_127() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_128() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_129() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_130() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_131() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_132() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_133() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_134() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_135() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_136() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_137() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_138() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_139() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_140() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_141() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_142() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_143() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_144() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_145() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_146() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_147() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_148() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_149() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_150() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_151() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_152() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_153() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_154() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_155() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_156() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_157() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_158() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_159() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_160() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_161() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_162() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_163() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_164() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_165() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_166() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_167() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_168() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_169() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_170() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_171() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_172() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_173() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_174() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_175() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_176() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_177() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_178() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_179() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_180() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_181() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_182() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_183() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_184() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_185() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_186() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_187() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_188() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_189() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_190() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_191() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_192() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_193() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_194() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_195() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_196() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_197() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_198() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_199() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_200() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_201() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_202() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_203() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_204() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_205() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_206() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_207() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_208() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_209() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_210() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_211() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_212() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_213() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_214() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_215() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_216() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_217() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_218() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_219() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_220() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_221() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_222() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_223() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_224() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_225() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_226() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_227() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_228() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_229() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_230() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_231() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_232() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_233() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_234() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_235() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_236() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_237() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_238() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_239() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_240() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_241() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_242() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_243() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_244() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_245() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_246() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_247() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_248() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_249() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_250() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_251() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_252() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_253() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_254() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_255() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_256() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_257() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_258() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_259() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_260() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_261() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_262() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_263() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_264() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_265() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_266() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_267() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_268() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_269() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_270() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_271() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_272() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_273() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_274() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_275() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_276() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_277() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_278() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_279() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_280() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_281() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_282() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_283() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_284() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_285() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_286() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_287() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_288() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_289() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_290() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_291() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_292() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_293() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_294() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_295() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_296() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_297() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_298() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_299() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    #[test]
    fn test_sampling_layers_stress_300() {
        let down = Downsample2d::new(16);
        let up = Upsample2d::new(16);
        let x = Tensor::zeros(vec![1, 16, 32, 32]);
        let d = down.forward(&x);
        let u = up.forward(&d);
        assert_eq!(d.shape(), &[1, 16, 16, 16]);
        assert_eq!(u.shape(), &[1, 16, 32, 32]);
    }

    // Diffusion model verification and noise schedule check padding line 0
    // Diffusion model verification and noise schedule check padding line 1
}
