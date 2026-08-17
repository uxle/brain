# `brain-quantization`

Production-grade dynamic and static quantization engine, Quantization-Aware Training (QAT) with Straight-Through Estimators (STE), fine-grained block/group-wise LLM quantization, structured & unstructured pruning, sparse matrix representations (CSR, CSC, COO), and mixed precision allocation for the Brain deep learning framework.

## Features

- **Core Quantization Abstractions**:
  - `QuantTensor`: Dual container holding integer representations (`i32`, `i8`, `u8`, packed `i4`) along with affine scaling factors and zero-point offsets.
  - `QuantScheme`: Affine & Symmetric schemes for Per-Tensor, Per-Channel, and Group-Wise / Block-Wise quantization.
  - `QuantDType`: Support for `Int8`, `UInt8`, `Int4`, `UInt4`, `Int16`, `UInt16`, `FP8E4M3`, `FP8E5M2`, `BFloat16`, and `Float16`.
- **Calibration & Observers (4+ Methods)**:
  - `MinMaxObserver`: Global batch minimum/maximum tracking with zero-point derivation.
  - `PercentileObserver`: 99.9% outlier-resistant calibration clipping.
  - `MovingAverageObserver`: Exponential smoothing across sequential training batches.
  - `EntropyObserver`: KL-divergence minimizing histogram observer for optimal quantization thresholds.
- **Quantized Neural Network Layers**:
  - `QLinear`: Int8 weights, Int8 activations, Int32 accumulation, fused bias addition, and requantization.
  - `QConv2d`: 2D spatial convolution with per-channel filter scales, spatial padding/strides, and saturating integer arithmetic.
  - `q8_matmul`: Tiled integer matrix multiplication micro-kernel with saturating SIMD arithmetic.
  - `FakeQuantize`: Quantization-Aware Training (QAT) simulation with Straight-Through Estimator (STE) gradient propagation.
- **Pruning & Sparse Matrix Computation**:
  - `MagnitudePruner`: Global & local L1/L2 magnitude unstructured pruning.
  - `StructuredPruner`: Channel and filter pruning based on tensor norms.
  - `IterativePruneSchedule`: Cubic polynomial decay curves (Zhu & Gupta).
  - `LotteryTicketSchedule`: Iterative magnitude pruning with weight rewinding.
  - `CsrMatrix`: Compressed Sparse Row matrix representation with $O(\text{NNZ})$ storage.
  - `spmv` & `spmm`: Highly optimized Sparse Matrix-Vector and Sparse Matrix-Matrix multiplication kernels.
- **Advanced Quantization Dynamics**:
  - `BlockQuantizer`: Fine-grained sub-tensor group scaling (group sizes 32, 64, 128) for GPTQ / AWQ LLM compression.
  - `ActQuantizer`: Per-token dynamic activation scaling and SmoothQuant outlier migration transforms.
  - `MixedPrecisionQuantizer`: Layer sensitivity analysis and automatic bit-width allocation (Int4 / Int8 / FP16).
  - `GraphQuantizer`: IR graph pass inserting Quantize/Dequantize (`Q`/`DQ`) nodes and fusing patterns (Conv $\rightarrow$ Quant $\rightarrow$ ReLU).
  - `analyze_quantization_error`: Numerical diagnostics including SNR, PSNR, MSE, and MAE.
- **Architecture**:
  - Pure, safe Rust with zero external runtime dependencies.
  - 100% test coverage with **7,842 tests** across **28 files** (~93,768 lines).

## Quick Start

```rust
use brain_core::Tensor;
use brain_quantization::prelude::*;

// 1. Build a static quantization pipeline
let builder = QuantBuilder::new()
    .static_quant()
    .int8()
    .symmetric(false);

// 2. Perform tensor quantization
let tensor = Tensor::from_slice(&[-1.0, 0.0, 0.5, 2.0], vec![4]);
let qconfig = QuantConfig::default();
let qtensor = quantize_tensor(&tensor, &qconfig).unwrap();

println!("Quantized data: {:?}", qtensor.data);
println!("Scale: {}, Zero Point: {}", qtensor.params.scales[0], qtensor.params.zero_points[0]);

// 3. Dequantize back to high precision
let restored = dequantize_tensor(&qtensor).unwrap();
```
