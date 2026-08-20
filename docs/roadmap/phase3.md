# Phase 3: Audit & De-Duplicate Tests in `brain-nn` + Layer Gradient Cross-Checks

**Stage:** A — Test Audit & De-duplication
**Status:** ✅ Complete
**Pass Rate:** 13 / 13 layer check tests passed

## Objective
Verify parameter gradients (`check_param_gradient`) for learnable layers, exact BatchNorm2d analytic backward formula, and initialization variance schemes.

## Mathematical Formulation
BatchNorm2d exact gradient w.r.t input:
$$
\frac{\partial L}{\partial x_i} = \frac{1}{N\sigma}\left[N \frac{\partial L}{\partial \hat{x}_i} - \sum_j \frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i \sum_j \frac{\partial L}{\partial \hat{x}_j}\hat{x}_j\right]
$$

## Key Verifications
1. **Layers Checked**: `Linear`, `Conv2d`, `Conv1d`, `ConvTranspose2d`, `BatchNorm2d`, `LayerNorm`, `RMSNorm`, `Embedding`, `Dropout`, `MaxPool2d`, `MultiheadAttention`.
2. **Embedding Scatter Accumulation**: Duplicate indices in batch sum gradients correctly.
3. **Weight Initialization**: Kaiming Normal variance matches $\frac{2}{fan\_in}$, Xavier Uniform bound matches $\sqrt{\frac{6}{fan\_in + fan\_out}}$.
4. **Parameter Completeness**: Sequential container returns all sub-module parameters.
