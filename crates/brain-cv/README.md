# `brain-cv`

Computer vision suite — convolutions, detection, segmentation, backbones, pooling, and augmentations — in 100% safe, zero-dependency Rust.

## Overview

`brain-cv` provides the vision primitives of the Brain ecosystem on top of `brain-core` tensors: seven convolution variants (standard, transposed, grouped, depthwise-separable, deformable, weight-standardized, ghost), multi-scale detection (`AnchorGenerator`, `DetectionHead`, NMS, `RoIAlign`), segmentation (`FcnHead`, `SegMetrics`, losses), backbone/FPN feature extraction (`BackboneZoo` with ResNet-50/MobileNet-V3 presets, `Fpn`), 2D pooling, and image augmentations (color jitter, geometric, mixup/cutmix, photometric, bbox transforms). Vectorized box math, affine grids, grid sampling, and histogram equalization live in `ops`.

## Features

- **Convolutions**: `Conv2d` (with `Conv2dConfig`), `ConvTranspose2d`, `GroupedConv2d`, `DepthwiseSeparableConv2d`, `DeformableConv2d`, `GhostModule`, plus residual/WS variants in `conv/`.
- **Detection**: `AnchorGenerator` (grid anchors), `DetectionHead`, `NmsConfig` + `non_max_suppression`, `RoIAlign`, `smooth_l1_loss`, box ops (`box_area`, `box_iou_matrix`), and postprocessing.
- **Segmentation**: `FcnHead`, `SegLossConfig`, `SegMetrics` (mIoU, pixel accuracy, dice).
- **Features**: `BackboneZoo` (`resnet50`, `mobilenet_v3`, `extract_features`), `Fpn` multi-scale fusion.
- **Pooling**: `AvgPool2d`, `MaxPool2d`.
- **Augmentation**: `Compose` pipeline, `ColorJitter`, geometric transforms, `mixup`/`cutmix`, `solarize`/`equalize`, `transform_bounding_boxes`.
- **Ops**: `affine_grid`, `grid_sample`, `equalize_histogram`.

## Modules

| Module | Contents |
|---|---|
| `conv/` | `mod` (`Conv2d`), `transposed`, `grouped`, `depthwise`, `deformable`, `ws`, `ghost`, `residual` |
| `detection/` | `anchor`, `head`, `nms`, `roi`, `roi_align`, `losses`, `postprocess` |
| `segmentation/` | `fcn`, `losses`, `metrics` |
| `feature/` | `backbones` (`BackboneZoo`), `fpn` |
| `augmentation/` | `color`, `geom`, `mix`, `photo`, `boxes` |
| `pooling/` | `AvgPool2d`, `MaxPool2d` |
| `ops/` | `boxes`, `geometry` (affine grid/grid sample), `hist_eq` |

## Quick Start

```rust
use brain_cv::prelude::*;

let conv = Conv2d::new(3, 16, 3);
let input = Tensor::zeros(vec![1, 3, 32, 32]);
let output = conv.forward(&input);
assert_eq!(output.shape()[1], 16);
```

## Testing

```bash
cargo test -p brain-cv -j 2
```

## Workspace Role

Core computer-vision building blocks for the Brain ecosystem. Depends only on `brain-core` (tensors) — zero external dependencies, 100% safe Rust.