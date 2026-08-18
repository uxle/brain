# Brain Ecosystem Crate Triage (1.0 Release)

This document categorizes all 33 crates in the repository by their 1.0 readiness status.

## Tier 1 — Production 1.0 Core (Fully Tested, Differentiable & Verified)
These crates form the stable, verified foundation of Brain 1.0 and are covered by automated regression and gradient checking suites:
1. `brain-core`: Core tensors, shape algebra, memory management, and cache-blocked GEMM.
2. `brain-autograd`: Reverse-mode automatic differentiation with deep-graph drop safety, VJPs, and tape execution.
3. `brain-nn`: Neural network modules (Linear, Conv2d, ConvTranspose2d, Embedding, BatchNorm2d, LayerNorm, Sequential).
4. `brain-loss`: Differentiable and stable loss functions (CrossEntropy, MSE, BCE, SmoothL1).
5. `brain-optim`: Parameter optimizers (SGD, Adam, AdamW), learning rate schedulers, and binary/text state dicts.
6. `brain-train`: End-to-end trainer engine, mini-batching, gradient accumulation, and checkpoint management.
7. `brain-onnx`: Pure-Rust ONNX protobuf parser, IR lowering, graph verification, and interpreter.
8. `brain-quantization`: 8-bit dynamic integer quantization, dequantization, and magnitude pruning.
9. `brain-cli`: Full command line interface (`make`, `check`, `run`, `train`, `doctor`).
10. `brain`: Top-level facade crate and CLI binary.
11. `brain-data` / `brain-dataset`: Dataset parsing, format ingestion, and batching fixtures.
12. `brain-export`: Model export serialization utilities.
13. `brain-graph`: Static computational graph representation and lowering passes.
14. `brain-metric`: Evaluation metrics (accuracy, precision, recall, F1-score, MSE).
15. `brain-utils`: Common hashing, timing, and utility routines.

---

## Tier 2 — Experimental / Domain Crates (In Active Development)
These crates provide experimental research abstractions and domain-specific models:
- `brain-transformer`, `brain-vit`, `brain-rnn`
- `brain-diffusion`, `brain-gan`
- `brain-rl`, `brain-gnn`, `brain-neuroevolution`
- `brain-cv`, `brain-text`, `brain-audio`
- `brain-federated`, `brain-distributed`
- `brain-compile`, `brain-benchmark`
