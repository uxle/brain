//! # Spatial Pooling Layers
//!
//! 2D MaxPooling, AveragePooling, and AdaptiveAveragePooling operations.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};

/// 2D Max Pooling layer.
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl MaxPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self { kernel_size, stride }
    }

    pub fn forward_tensor(&self, input: &Tensor) -> Tensor {
        brain_core::tensor::pool::max_pool2d(
            input,
            (self.kernel_size, self.kernel_size),
            (self.stride, self.stride),
            (0, 0),
        )
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        Ok(self.forward_tensor(input))
    }
}

/// 2D Average Pooling layer.
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self { kernel_size, stride }
    }

    pub fn forward_tensor(&self, input: &Tensor) -> Tensor {
        brain_core::tensor::pool::avg_pool2d(
            input,
            (self.kernel_size, self.kernel_size),
            (self.stride, self.stride),
            (0, 0),
        )
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        Ok(self.forward_tensor(input))
    }
}

/// 2D Adaptive Average Pooling layer to a fixed output size.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveAvgPool2d {
    pub out_h: usize,
    pub out_w: usize,
}

impl AdaptiveAvgPool2d {
    pub fn new(out_h: usize, out_w: usize) -> Self {
        Self { out_h, out_w }
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        Ok(brain_core::tensor::pool::adaptive_avg_pool2d(input, self.out_h, self.out_w))
    }
}

/// 2D Adaptive Max Pooling layer to a fixed output size.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveMaxPool2d {
    pub out_h: usize,
    pub out_w: usize,
}

impl AdaptiveMaxPool2d {
    pub fn new(out_h: usize, out_w: usize) -> Self {
        Self { out_h, out_w }
    }
}

impl Module for AdaptiveMaxPool2d {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        Ok(brain_core::tensor::pool::adaptive_max_pool2d(input, self.out_h, self.out_w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_max_pool2d_downsampling() {
        let mp = MaxPool2d::new(2, 2);
        let t = Tensor::from_slice(
            &[
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![1, 1, 4, 4],
        );
        let out = mp.forward(&t).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.to_vec(), vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_avg_pool2d_downsampling() {
        let ap = AvgPool2d::new(2, 2);
        let t = Tensor::from_slice(
            &[
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![1, 1, 4, 4],
        );
        let out = ap.forward(&t).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.to_vec(), vec![3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn test_pool_multiple_channels() {
        let mp = MaxPool2d::new(2, 2);
        let t = Tensor::zeros(vec![2, 3, 6, 6]);
        let out = mp.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 3, 3, 3]);
    }

    #[test]
    fn test_adaptive_pool_layers() {
        let ap = AdaptiveAvgPool2d::new(2, 2);
        let t = Tensor::from_slice(
            &[
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![1, 1, 4, 4],
        );
        let out = ap.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![3.5, 5.5, 11.5, 13.5]);

        let mp = AdaptiveMaxPool2d::new(2, 2);
        let out = mp.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![6.0, 8.0, 14.0, 16.0]);

        let g = AdaptiveAvgPool2d::new(1, 1);
        let out = g.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![8.5]);
    }
}
