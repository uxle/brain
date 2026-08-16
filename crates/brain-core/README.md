# `brain-core` (v0.2.0)

> Foundational N-dimensional Tensor Engine, Stride Views, SIMD Kernels, and Mathematical Operations for the Brain Deep Learning Framework.

## Overview

`brain-core` provides the low-level foundation for the entire Brain framework in 100% pure, safe, dependency-free Rust. It implements N-dimensional tensors with arbitrary striding, zero-copy tensor slicing and views, cache-optimized matrix multiplication kernels, FFT, BLAS level 1-3 routines, random number generators, memory management, and serialization.

## Architecture

| Module | Description |
|---|---|
| `tensor` | Core `Tensor` struct with shape, strides, storage, and zero-copy view transformations |
| `blas` | Level 1 (dot, axpy, norm), Level 2 (gemv, ger), and Level 3 (gemm, syrk) kernels |
| `math` | Element-wise arithmetic, unary transcendental ops, trigonometry, and activations |
| `linalg` | Matrix inversion, LU decomposition, QR decomposition, Cholesky, SVD, eigenvalues |
| `reduction` | Sum, mean, variance, standard deviation, argmax, argmin along arbitrary dimensions |
| `broadcast` | Multi-dimensional broadcasting rules adhering to NumPy semantics |
| `indexing` | Slice, select, narrow, gather, scatter, mask, and index_select operations |
| `random` | Deterministic PRNGs: Uniform, Normal (Box-Muller), Bernoulli, Xavier, Kaiming |
| `conv_pool` | 1D/2D/3D convolution and pooling forward/backward reference kernels |
| `fft` | 1D/2D Fast Fourier Transforms and Inverse FFTs (Cooley-Tukey radix-2) |
| `sparse` | Compressed Sparse Row (CSR) and Coordinate (COO) sparse matrix representations |
| `serialization` | Binary and endian-aware tensor serialization and deserialization |
| `simd` | Explicit vectorization and lane-aligned memory arithmetic helpers |

## Quick Start

```rust
use brain_core::Tensor;

fn main() {
    // Create tensors with arbitrary shapes
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    // Perform matrix multiplication
    let c = a.matmul(&b);
    println!("Shape: {:?}, Data: {:?}", c.shape(), c.to_vec());

    // Zero-copy slicing & views
    let view = c.view(vec![4, 1]);
    println!("Reshaped: {:?}", view.shape());
}
```

## Quality & Verification

- **Tests**: 10,851 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-core -- -D warnings`)
- **Dependencies**: `std` only
