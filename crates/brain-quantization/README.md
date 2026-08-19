# `brain-quantization`

Pure-Rust quantization and sparsity engine: dynamic/static int8, QAT, pruning, and sparse linear algebra.

## Overview

`brain-quantization` provides post-training dynamic and static quantization, fake-quantization for quantization-aware training, magnitude and structured pruning, CSR sparse-matrix kernels, block/activation/mixed-precision quantizers, and end-to-end graph quantization — all over `brain-core` tensors in 100% safe Rust.

## Features

- **Core quantization** — `QuantTensor`, `QParams`, `QuantScheme` (symmetric/affine), `QuantDType` (Int8 etc.), `quantize_tensor` / `dequantize_tensor` / `apply_magnitude_prune` helpers.
- **Dynamic & static** — `DynamicQuantizer`, `StaticQuantizer`, `FakeQuantize` (QAT), and calibration observers (`MinMaxObserver`, `PercentileObserver`, `MovingAverageObserver`, `EntropyObserver`).
- **Quantized operators** — `QLinear` (with `QLinearConfig`), `QConv2d`, `q8_matmul` (`QMatMulConfig`).
- **Pruning** — `MagnitudePruner`, `StructuredPruner`, `IterativePruneSchedule`, and CSR/CSC/COO sparse matrices with `spmm` / `spmv`.
- **Advanced quantizers** — `BlockQuantizer`, `ActQuantizer` (activation quantization), `MixedPrecisionQuantizer`, `GraphQuantizer` (whole-graph).
- **Analysis & runtime** — `analyze_quantization_error` (`QuantErrorReport`), `QuantBench` (`QuantBenchReport`), and a `QuantRuntime` plus `QuantBuilder` pipeline (`PipelineMode`).

## Modules

| Module | Description |
|---|---|
| `core` | `QuantTensor`, `QParams`, `QuantDType`, `QuantScheme`, error types |
| `quantizer` / `quant_dynamic` / `quant_static` / `fake_quant` | Quantizer implementations and QAT |
| `calibration` | Min/max, percentile, moving-average, entropy observers |
| `qlinear` / `qconv` / `qmatmul` | Quantized operators |
| `prune` | Magnitude and structured pruners, iterative schedules |
| `sparse` | CSR/CSC/COO matrices and sparse kernels |
| `block_quant` / `act_quant` / `mixed` / `graph_quant` | Advanced quantization strategies |
| `error_analysis` / `bench_quant` | Error analysis and benchmark reports |
| `runtime` / `builder` | Quantization runtime and pipeline builder |
| `impl` | `quantize_tensor`, `dequantize_tensor`, `apply_magnitude_prune` |

## Quick Start

```rust
use brain_core::Tensor;
use brain_core::tensor::arithmetic::matmul;
use brain_quantization::{dequantize_tensor, quantize_tensor, QuantConfig, QuantDType};

let weight = Tensor::from_vec(vec![0.12, -0.45, 0.78, -0.23, 0.91, -0.05, 0.33, -0.88], vec![2, 4]);
let cfg = QuantConfig { dtype: QuantDType::Int8, ..Default::default() };
let qweight = quantize_tensor(&weight, &cfg).unwrap();
let deq = dequantize_tensor(&qweight).unwrap();

let x = Tensor::from_vec(vec![1.0, 0.5, -0.5, 2.0], vec![1, 4]);
let y = matmul(&x, &deq.transpose(0, 1)); // quantized linear forward
```

## Testing

```bash
cargo test -p brain-quantization --test quant_linear -j 2
cargo test -p brain-quantization -j 2
```

`quant_linear` verifies fp32 vs quantized linear outputs agree within `1e-2`.

## Workspace Role

Depends on `brain-core` and `brain-graph`. Consumer: the `brain` facade (via its `export` feature).