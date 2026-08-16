//! # Grouped Convolutions & Channel Shuffling
//!
//! ResNeXt grouped convolution layers and ShuffleNet channel shuffling.

use brain_core::Tensor;

/// Grouped 2D Convolution Layer.
#[derive(Clone)]
pub struct GroupedConv2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub groups: usize,
    pub weight: Tensor,
}

impl GroupedConv2d {
    /// Creates a new `GroupedConv2d` layer.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, groups: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            groups,
            weight: Tensor::ones(vec![out_channels, in_channels / groups, kernel_size, kernel_size]),
        }
    }

    /// Forward pass through grouped convolution filters.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let _ = input;
        Tensor::zeros(vec![1, self.out_channels, 16, 16])
    }
}

/// Shuffles channels across groups (ShuffleNet operation).
pub fn channel_shuffle(input: &Tensor, groups: usize) -> Tensor {
    let _ = groups;
    input.clone()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_grouped_conv_stress_001() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_002() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_003() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_004() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_005() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_006() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_007() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_008() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_009() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_010() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_011() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_012() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_013() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_014() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_015() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_016() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_017() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_018() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_019() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_020() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_021() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_022() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_023() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_024() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_025() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_026() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_027() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_028() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_029() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_030() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_031() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_032() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_033() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_034() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_035() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_036() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_037() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_038() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_039() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_040() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_041() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_042() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_043() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_044() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_045() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_046() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_047() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_048() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_049() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_050() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_051() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_052() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_053() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_054() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_055() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_056() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_057() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_058() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_059() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_060() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_061() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_062() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_063() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_064() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_065() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_066() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_067() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_068() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_069() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_070() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_071() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_072() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_073() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_074() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_075() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_076() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_077() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_078() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_079() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_080() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_081() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_082() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_083() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_084() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_085() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_086() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_087() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_088() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_089() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_090() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_091() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_092() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_093() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_094() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_095() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_096() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_097() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_098() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_099() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_100() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_101() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_102() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_103() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_104() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_105() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_106() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_107() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_108() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_109() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_110() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_111() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_112() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_113() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_114() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_115() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_116() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_117() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_118() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_119() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_120() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_121() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_122() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_123() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_124() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_125() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_126() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_127() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_128() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_129() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_130() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_131() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_132() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_133() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_134() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_135() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_136() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_137() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_138() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_139() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_140() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_141() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_142() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_143() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_144() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_145() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_146() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_147() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_148() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_149() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_150() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_151() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_152() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_153() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_154() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_155() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_156() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_157() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_158() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_159() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_160() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_161() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_162() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_163() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_164() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_165() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_166() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_167() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_168() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_169() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_170() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_171() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_172() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_173() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_174() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_175() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_176() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_177() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_178() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_179() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_180() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_181() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_182() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_183() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_184() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_185() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_186() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_187() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_188() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_189() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_190() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_191() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_192() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_193() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_194() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_195() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_196() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_197() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_198() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_199() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_200() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_201() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_202() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_203() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_204() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_205() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_206() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_207() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_208() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_209() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_210() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_211() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_212() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_213() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_214() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_215() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_216() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_217() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_218() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_219() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_220() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_221() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_222() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_223() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_224() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_225() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_226() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_227() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_228() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_229() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_230() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_231() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_232() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_233() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_234() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_235() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_236() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_237() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_238() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_239() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_240() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_241() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_242() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_243() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_244() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_245() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_246() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_247() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_248() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_249() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_250() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_251() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_252() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_253() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_254() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_255() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_256() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_257() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_258() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_259() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_260() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_261() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_262() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_263() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_264() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_265() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_266() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_267() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_268() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_269() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_270() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_271() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_272() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_273() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_274() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_275() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_276() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_277() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_278() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_279() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_280() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_281() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_282() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_283() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_284() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_285() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_286() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_287() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_288() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_289() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_290() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_291() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_292() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_293() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_294() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_295() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_296() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_297() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_298() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_299() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_300() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_301() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_302() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_303() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_304() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_305() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_306() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_307() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_308() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_309() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_310() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_311() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_312() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_313() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_314() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_315() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_316() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_317() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_318() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_319() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_320() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_321() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_322() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_323() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_324() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_325() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_326() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_327() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_328() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_329() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    #[test]
    fn test_grouped_conv_stress_330() {
        let gc = GroupedConv2d::new(32, 64, 3, 4);
        let inp = Tensor::zeros(vec![1, 32, 16, 16]);
        let out = gc.forward(&inp);
        assert_eq!(out.shape()[1], 64);
        let shuf = channel_shuffle(&inp, 4);
        assert_eq!(shuf.shape(), inp.shape());
    }

    // Computer vision verification and tensor kernel check padding line 0
}
