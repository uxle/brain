# `brain-core`

Zero-dependency N-dimensional tensor engine and computational foundation of the Brain deep learning framework.

## Overview

`brain-core` is the bottom layer of the workspace: a pure, 100% safe, `std`-only Rust tensor library. It provides an N-dimensional `Tensor` with strided views, SIMD-friendly arithmetic kernels, cache-blocked GEMM, BLAS level 1–3 routines, linear algebra decompositions, FFT, sparse matrices, deterministic PRNGs, and binary checkpoint serialization. Every other crate in the workspace builds on it.

## Features

- **Tensor primitives** — `Tensor::from_vec`, `from_slice`, `zeros`, `ones`, `arange`, `eye`, `rand`, `randn`; strided `view`s, `reshape`, `transpose`, `slice_multi`, `gather`, `scatter`, `scatter_add`, `cat`, `stack`, `hstack`, `vstack`.
- **Arithmetic** — free-function `matmul` (cache-blocked), `matvec`, `outer`, `cosine_similarity`, element-wise math (`exp`, `log`, trig, `sqrt`, `rsqrt`, …) and fused ops.
- **Linear algebra** — `lu` / `lu_solve`, `qr` / `qr_solve`, `cholesky` / `cholesky_solve`, `svd_symmetric` / `svd_solve`, `eigh`.
- **Reductions & scan ops** — `sum`, `mean`, `var_mean`, `std`, argmax/argmin, `cumsum`, `cumprod`, `roll`, `repeat` along any dimension.
- **Neural primitives** — `conv2d`, `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`, `adaptive_max_pool2d`, `unfold` / `fold`.
- **BLAS kernels** — `axpy`, `dot`, `scal`, `nrm2`, `asum`, `iamax`, `rot`, `swap`, `copy`, `gemv`, `ger`, `gemm`.
- **SIMD helpers** — lane-aligned `simd_add`, `simd_sub`, `simd_mul`, `simd_fma`, `simd_dot`, `simd_sum`, `simd_relu` (see `tensor::simd`).
- **Fast transforms** — 1D radix-2 Cooley–Tukey `fft_radix2` and inverse transform.
- **Sparse** — CSR matrices with `spmm` sparse-dense multiplication (`tensor::sparse`).
- **Infrastructure** — device abstractions (`Device`, `CpuBackend`, `SimdCpuBackend`), dtype system (`F64`…`U8`, `Bool`), deterministic PRNGs (XORShift128+, PCG32, SplitMix64, ChaCha8), aligned memory pools, binary checkpoint format v2 with CRC32 (`BrainModelFile`, `TensorArchive`).
- **Experimental** — `BrainMind` biologically-inspired associative memory with token encoding, neural forward pass, and text generation.

## Modules

| Module | Description |
|---|---|
| `tensor` | N-D `Tensor` plus arithmetic, blas, linalg, reduction, conv, pool, indexing, simd, sparse, view, factory and more |
| `shape` | Shape algebra, strides, dimension broadcast resolution |
| `device` | Device/backend abstractions and thread-local device stack |
| `dtype` | Data types and promotion rules |
| `random` | Deterministic PRNGs and distributions |
| `memory` | Aligned buffers, arenas, buddy allocators, binned pools |
| `serialization` | Binary checkpoint format v2 with CRC32 integrity |
| `error` | Categorized `BrainError` / `BrainResult` with chained context |
| `brain_mind` | Associative-memory experiment (`BrainMind`, `TeachSummary`) |

## Quick Start

```rust
use brain_core::Tensor;
use brain_core::tensor::arithmetic::matmul;

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

let c = matmul(&a, &b);          // free function, NOT a method
println!("{:?}", c.shape());     // [2, 2]

let v = c.transpose(0, 1);       // strided view
println!("{:?}", v.to_vec());    // [5.0, 7.0, 6.0, 8.0, ...] row-major data
```

`matmul` (and `matvec`, `outer`, `cosine_similarity`) are free functions in `brain_core::tensor::arithmetic`; `Tensor` itself has no `matmul` method.

## Testing

```bash
cargo test -p brain-core --test numerical_check -j 2
cargo test -p brain-core -j 2
```

(The `-j 2` cap avoids CPU/memory starvation in this 33-crate workspace.)

## Workspace Role

`brain-core` has **zero dependencies** (`std` only). It is the base layer of the workspace: ~32 crates depend on it, including `brain-autograd`, `brain-nn`, `brain-onnx`, and `brain-quantization`.
