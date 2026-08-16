# `brain-cv` (v0.2.0)

> Computer Vision Architectures, 2D/3D Convolutions, Detection Heads, Anchor Generation, NMS, and Augmentations.

## Overview

`brain-cv` delivers complete computer vision capabilities. It implements 2D/3D convolutions, deformable convolutions, depthwise-separable convolutions, RoI pooling, RoI Align, Non-Maximum Suppression (NMS), anchor boxes, FPN feature pyramids, semantic segmentation heads, and spatial image transforms.

## Architecture

| Module | Description |
|---|---|
| `conv` | 2D/3D Conv, Transposed Conv, Deformable Conv, Depthwise-Separable Conv |
| `detection` | Anchor generators, IoU matchers, bounding box encoding/decoding, Fast NMS, RoI Align |
| `segmentation`| Semantic segmentation heads, FCN, U-Net decoder blocks, feature pyramid networks |
| `augmentation` | Color jitter, affine transforms, random cropping, rotation, horizontal/vertical flips |
| `feature` | Classical feature extractors: Sobel, Laplacian, Canny, Harris corner response |

## Quick Start

```rust
use brain_cv::detection::nms_fast;
use brain_core::Tensor;

fn main() {
    let boxes = Tensor::from_vec(vec![
        0.0, 0.0, 10.0, 10.0,
        1.0, 1.0, 9.0, 9.0,
        50.0, 50.0, 100.0, 100.0,
    ], vec![3, 4]);
    let scores = vec![0.9, 0.8, 0.95];

    let kept = nms_fast(&boxes, &scores, 0.5);
    println!("Kept box indices: {:?}", kept);
}
```

## Quality & Verification

- **Tests**: 14,921 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-cv -- -D warnings`)
- **Dependencies**: `std` + `brain-core`
