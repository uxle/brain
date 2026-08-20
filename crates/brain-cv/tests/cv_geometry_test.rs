//! Tests for Computer Vision bounding boxes, IoU, and NMS
use brain_core::Tensor;
use brain_cv::*;

#[test]
fn test_cv_conv2d_and_pooling() {
    let conv = Conv2d::new(3, 8, 3);
    let input = Tensor::zeros(vec![1, 3, 16, 16]);
    let output = conv.forward(&input);
    assert_eq!(output.shape()[1], 8);

    let pool = MaxPool2d::new(2, 2);
    let pooled = pool.forward(&output);
    assert_eq!(pooled.shape()[2], output.shape()[2] / 2);
}
