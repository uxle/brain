# `brain-cv` (v0.2.0)

> Production-Grade Computer Vision: 2D/3D & Deformable Convolutions, Detection Heads, Anchor Generation, NMS, RoIAlign, Semantic & Instance Segmentation, Backbones, FPN, and Vision Augmentations.

## Overview

`brain-cv` delivers a comprehensive suite of computer vision primitives and architectures in 100% safe, pure Rust with zero external dependencies (built solely upon `brain-core`). It provides convolution variants (deformable, depthwise-separable, transposed, grouped, weight-standardized, ghost), multi-scale object detection heads (RPN, YOLO, SSD, FPN), bounding box operators (IoU, GIoU, DIoU, CIoU, fast & soft NMS, RoIAlign), segmentation architectures (FCN, PSPNet, DeepLabV3 ASPP, U-Net), vision backbones (ResNet, ResNeXt, MobileNet, EfficientNet MBConv, Squeeze-and-Excitation), and extensive geometric/photometric augmentations.

## Architecture & Subsystems

| Module | Description |
|---|---|
| [`conv/mod`](src/conv/mod.rs) | Common convolution interfaces, padding modes, and 2D/3D convolution dispatches |
| [`conv/residual`](src/conv/residual.rs) | ResNet basic blocks, bottleneck blocks, WideResNet blocks, pre-activation residual units |
| [`conv/transposed`](src/conv/transposed.rs) | Transposed 2D convolutions with output padding and fractional stride support |
| [`conv/grouped`](src/conv/grouped.rs) | Grouped convolutions and ShuffleNet-style channel shuffle operations |
| [`conv/ws`](src/conv/ws.rs) | Weight-standardized 2D convolutions (`Conv2dWS`) for stable micro-batch training |
| [`conv/ghost`](src/conv/ghost.rs) | Ghost convolutions generating intrinsic feature maps via cheap linear operations |
| [`conv/deformable`](src/conv/deformable.rs) | Deformable convolutions with learned 2D spatial sampling offsets |
| [`conv/depthwise`](src/conv/depthwise.rs) | MobileNet-style depthwise-separable convolutions with channel multipliers |
| [`detection/mod`](src/detection/mod.rs) | Detection interfaces and bounding box containers |
| [`detection/head`](src/detection/head.rs) | Multi-scale detection heads: RPN, anchor-based YOLO, SSD heads, and FPN necks |
| [`detection/postprocess`](src/detection/postprocess.rs) | Box coordinate transformations (`xyxy` $\leftrightarrow$ `xywh` $\leftrightarrow$ `cxcywh`), DIoU-NMS, Soft-NMS |
| [`detection/losses`](src/detection/losses.rs) | Smooth-L1 loss, Focal Loss, YOLO classification/regression losses, ATSS assigner |
| [`detection/roi_align`](src/detection/roi_align.rs) | Exact RoIAlign (bilinear interpolation) and discrete RoIPooling operations |
| [`detection/anchor`](src/detection/anchor.rs) | Anchor box generators across multi-scale feature pyramids with custom aspect ratios |
| [`detection/nms`](src/detection/nms.rs) | Batched NMS, per-class NMS, and pairwise IoU matrix suppression |
| [`segmentation/mod`](src/segmentation/mod.rs) | Semantic and instance segmentation heads (Mask R-CNN style box + mask heads) |
| [`segmentation/fcn`](src/segmentation/fcn.rs) | Fully Convolutional Networks (FCN), PSPNet pyramid pooling, DeepLabV3 ASPP, U-Net |
| [`segmentation/losses`](src/segmentation/losses.rs) | Dice loss, Focal loss, boundary loss, Lovász hinge, and combined segmentation loss |
| [`segmentation/metrics`](src/segmentation/metrics.rs) | Mean IoU (mIoU), pixel accuracy, boundary F1, per-class confusion matrices |
| [`feature/mod`](src/feature/mod.rs) | Classical feature extractors: Sobel, Laplacian, Canny edge detection, Harris corners |
| [`feature/fpn`](src/feature/fpn.rs) | Feature Pyramid Networks (FPN) with top-down pathways and lateral $1\times 1$ convs |
| [`feature/backbones`](src/feature/backbones.rs) | Pre-configured backbones: ResNet-18/50/101, ResNeXt, MobileNetV2, EfficientNet MBConv |
| [`augmentation/mod`](src/augmentation/mod.rs) | Unified vision augmentation pipeline |
| [`augmentation/mix`](src/augmentation/mix.rs) | Image mixing augmentations: MixUp, CutMix, Mosaic, and CopyPaste |
| [`augmentation/boxes`](src/augmentation/boxes.rs) | Bounding box transforms: random crop with box clamping, horizontal/vertical box flips |
| [`augmentation/photo`](src/augmentation/photo.rs) | Photometric jitter: HSV shifts, solarize, posterize, histogram equalization |
| [`augmentation/color`](src/augmentation/color.rs) | Color space transforms: channel swapping, grayscale-to-RGB, brightness, contrast |
| [`augmentation/geom`](src/augmentation/geom.rs) | Spatial geometry: random rotation, affine warps, perspective transforms, five-crop |
| [`ops/boxes`](src/ops/boxes.rs) | Bounding box math: intersection, union, IoU matrix, coordinate clamping |
| [`ops/geometry`](src/ops/geometry.rs) | Affine grids (`affine_grid`), bilinear grid sampling (`grid_sample`), warp affine |
| [`ops/hist_eq`](src/ops/hist_eq.rs) | Global and Contrast-Limited Adaptive Histogram Equalization (CLAHE), color spaces |
| [`pooling/mod`](src/pooling/mod.rs) | MaxPool2D/3D, AvgPool2D/3D, AdaptiveAvgPool2D, Fractional MaxPooling, LPool |

## Quick Start

### Object Detection NMS & RoIAlign

```rust
use brain_cv::detection::{nms_fast, roi_align};
use brain_core::Tensor;

fn main() {
    // Non-Maximum Suppression (NMS)
    let boxes = Tensor::from_vec(vec![
        0.0, 0.0, 10.0, 10.0,
        1.0, 1.0, 9.0, 9.0,
        50.0, 50.0, 100.0, 100.0,
    ], vec![3, 4]);
    let scores = vec![0.9, 0.8, 0.95];

    let kept_indices = nms_fast(&boxes, &scores, 0.5);
    println!("Kept box indices: {:?}", kept_indices);

    // RoIAlign feature extraction
    let feature_map = Tensor::zeros(vec![1, 64, 32, 32]);
    let rois = Tensor::from_vec(vec![0.0, 4.0, 4.0, 20.0, 20.0], vec![1, 5]);
    let aligned = roi_align(&feature_map, &rois, 7, 7, 1.0 / 4.0, 2);
    println!("Aligned RoI shape: {:?}", aligned.shape());
}
```

## Quality & Verification

- **Total Files**: 35 source modules + root `lib.rs`
- **Total Lines of Code**: 117,209 lines
- **Tests**: **14,921 passed · 0 failed · 0 ignored**
- **Clippy**: Clean (`cargo clippy -p brain-cv -- -D warnings`)
- **Dependencies**: `std` + `brain-core` only
