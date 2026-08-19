# `brain-vit`

Vision Transformer (ViT) implementation with patch embeddings, backbone presets, training utilities, and detection/segmentation heads — in 100% safe, zero-dependency Rust.

## Overview

`brain-vit` builds complete ViT models from scratch on `brain-core`: convolutional `PatchEmbed` tokenization, learned/sinusoidal `PosEmbed` positional embeddings, stacked transformer blocks with classification/DINO-style heads, and CLS/mean pooling. It adds `VitBackbone` presets (micro/tiny/base), a `training` toolkit (optimizer, LR scheduler, early stopping, gradient clipping), detection (`Bbox`, `DetectionHead`, IoU, NMS) and segmentation (`SegDecoder`, mIoU, pixel accuracy) decoders, and `export` utilities (`JsonCheckpoint`, `OnnxStub`, `ModelCard`).

## Features

- **Model**: `VitModel` (in `impl`) with `forward(&pixels, batch)` and `predict` (softmax probabilities); `ViT` core module with `depth` transformer blocks; `save`/`load` checkpoint roundtrip.
- **Patch & position**: `PatchEmbed` (conv/unfold modes), `PosEmbed` (learned/sinusoidal, with or without CLS token).
- **Backbones**: `VitBackbone` + `BackboneSize` presets with `extract_cls`/feature extraction.
- **Training**: `Optimizer`/`OptimizerConfig`, `LrScheduler` (incl. cosine-with-warmup `ScheduleType`), `EarlyStopping`, `GradientClipper`.
- **Downstream**: `DetectionHead`, `Bbox`, `iou`, `nms`; `SegDecoder`, `per_class_iou`, `pixel_accuracy`.
- **Export**: `JsonCheckpoint`, `OnnxStub`, `ModelCard`.
- **Infrastructure**: `Tensor2D`/`Tensor3D` helpers, `SimpleRng`, `VitError`/`VitResult`, and constants for the ViT-Base configuration (`DEFAULT_IMAGE_SIZE` 224, `DEFAULT_PATCH_SIZE` 16, `DEFAULT_EMBED_DIM` 768, etc.).

## Modules

| Module | Contents |
|---|---|
| `patch/` | `PatchEmbed`, `pos_embed` (`PosEmbed`) |
| `vit/` | Transformer blocks, heads (classification/DINO), full `ViT` |
| `backbones.rs` | `VitBackbone`, `BackboneSize` presets |
| `training.rs` | `Optimizer`, `LrScheduler`, `EarlyStopping`, `GradientClipper` |
| `detection.rs` | `Bbox`, `DetectionHead`, `iou`, `nms` |
| `segmentation.rs` | `SegDecoder`, `per_class_iou`, `pixel_accuracy` |
| `export.rs` | `JsonCheckpoint`, `OnnxStub`, `ModelCard` |
| `core.rs`, `config.rs`, `ops.rs`, `utils.rs`, `impl.rs` | `VitConfig`, tensors, RNG, errors, `VitModel` |

## Quick Start

```rust
use brain_vit::{VitConfig, VitModel};

let cfg = VitConfig::default();
let mut model = VitModel::new(cfg, 42).unwrap();

let pixels = vec![0.0f64; 3 * 224 * 224]; // 1 image
let output = model.forward(&pixels, 1).unwrap();
println!("Logits: {:?}", &output.logits[0][..5]);
```

## Testing

```bash
cargo test -p brain-vit -j 2
```

## Workspace Role

Vision-Transformer model family for the Brain ecosystem, demonstrating a full training/inference lifecycle on `brain-core` alone. Depends only on `brain-core` — zero external dependencies, 100% safe Rust.