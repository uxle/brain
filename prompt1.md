# PROMPT 1 — UPGRADE `brain-core` TO MAXIMUM STRENGTH

## Mission

You are an elite Rust systems engineer specialized in high-performance deep learning
infrastructure. Upgrade the `brain-core` crate — the foundation of the Brain DL framework —
from its baseline state to **maximum strength**: production-grade performance, zero external
dependencies, broad API coverage, deep correctness, and complete test coverage.

## Current State (Baseline, 15 files, ~23K lines)

- `src/lib.rs` (1,283): crate root, re-exports, prelude, Config, type aliases
- `src/error.rs` (3,691): error types, macros, result helpers
- `src/dtype.rs` (3,257): DType enum, casts, DTypeInfo, DTypeMap
- `src/device.rs` (3,031): Device enum, DeviceList, DeviceProperties, guards
- `src/shape.rs` (3,393): Shape, dims, strides, broadcast helpers
- `src/memory.rs` (806): memory management
- `src/random.rs` (573): PRNG and distributions
- `src/serialization.rs` (573): save/load tensors (bin/JSON)
- `src/tensor/mod.rs` (603): TensorStats, iterators, index, layout, flags
- `src/tensor/impl.rs` (1,645): Tensor struct, constructors, accessors
- `src/tensor/arithmetic.rs` (1,013): add/sub/mul/div/matmul, broadcasting
- `src/tensor/math.rs` (1,213): exp/log/sqrt/sigmoid/tanh/pow, trig
- `src/tensor/linalg.rs` (946): det, inv, cholesky, QR, eig, SVD, norm
- `src/tensor/reduction.rs` (842): sum/mean/min/max/var/std/prod
- `src/tensor/indexing.rs` (614): indexing, slicing, gather/scatter

## Constraints

- Zero runtime dependencies (std-only; keep existing deps, add none).
- Edition 2021, stable Rust, no nightly features.
- Every source file **must be 3000–10000 lines** (docs + tests included).
- You **may add new files/modules** — encouraged to add as many as needed.
- Do not break the existing public API; the upgrade is additive and strengthening.

## A. NEW MODULES (create each as 3000–10000-line files)

1. `src/tensor/broadcast.rs` — numpy-style broadcast engine: lazy broadcast views (no
   copy), `BroadcastInfo` (offset/stride mapping), `broadcast_to`, `broadcast_batch`,
   `broadcast_unsqueeze/squeeze`, precise shape-conflict errors.
2. `src/tensor/blas.rs` — hand-optimized BLAS-level kernels: tiled/blocked `gemm` with
   loop unrolling and cache blocking, `gemv`, `axpy`, `dot`, `scal`, `ger`, `nrm2`, `asum`,
   `iamax`, rank-1/rank-k updates, multithreaded tiling via `std::thread::scope` when
   `max_threads > 1`.
3. `src/tensor/simd.rs` — autovectorized kernels + `std::arch` paths for x86_64
   (AVX2/SSE) and aarch64 (NEON): elementwise fma, dot, reductions (sum/max/min/mean),
   runtime detection (`is_x86_feature_detected!`), scalar fallback.
4. `src/tensor/view.rs` — zero-copy strided views: arbitrary/negative strides,
   `transpose`, `permute`, `narrow`, `flip`, `as_strided`, `contiguous()`,
   `is_contiguous()`, `make_contiguous()` with copy-on-write semantics, view chaining.
5. `src/tensor/special.rs` — numerically stable `logsumexp`, `softmax`, `log_softmax`,
   `softplus`, `gelu`, `silu`, `mish`, `hard_swish/sigmoid`, `erf`, `erfinv`, `gamma`,
   `lgamma`, `digamma`, `beta`, `bessel_i0/i1`, `clip`, `nan_to_num`, `isclose`, `where`.
6. `src/tensor/conv.rs` — 1D/2D/3D convolution (NCHW) with im2col, direct and tiled
   strategies; transposed, dilated, grouped, depthwise, pointwise conv; padding modes
   `valid`/`same`/`explicit`/circular; automatic kernel selection.
7. `src/tensor/pool.rs` — `max_pool1d/2d/3d`, `avg_pool1d/2d/3d`,
   `adaptive_max_pool/adaptive_avg_pool`, global pooling, stride/dilation/padding
   options, ceiling mode.
8. `src/tensor/pad.rs` — `pad` (constant/reflect/replicate/circular per-dim),
   `unfold`/`fold` (sliding window extract/reconstruct), `roll`, `rot90`,
   `repeat_interleave`, `tile`, `expand`.
9. `src/tensor/compare.rs` — `eq/ne/lt/le/gt/ge` (elementwise and scalar), logical
   `and/or/xor/not`, `argmax`, `argmin`, `topk`, `sort`, `argsort`, `unique`,
   `nonzero`, `masked_select`, `masked_fill`, `index_select`, `bincount`.
10. `src/tensor/fold.rs` — `einsum` with full subscript parser (ellipsis, broadcast
    dims, `ii->i` diagonals), `tensordot`, `kron`, `outer`, `inner`, `vdot`,
    `cumsum`, `cumprod`, `cummax`, `cummin` with axis.
11. `src/tensor/fft.rs` — pure-Rust FFT: radix-2 Cooley-Tukey, split-radix, Bluestein
    for arbitrary sizes, `fft/ifft/rfft/irfft/fft2d`, `fftshift`, window functions
    (hann, hamming, blackman, bartlett, kaiser).
12. `src/tensor/sparse.rs` — `SparseTensor` (COO + CSR): sparse add/mul, SpMV/SpMM,
    sparse-dense ops, `to_dense`, `from_dense`, `coalesce`, `sparse_eye`, `sparse_diag`.
13. `src/tensor/quant.rs` — int8/uint8 quantize/dequantize, per-tensor and per-channel
    scales, symmetric/asymmetric with zero-point, `q8_matmul` with saturation and
    rounding control.
14. `src/tensor/hist.rs` — `histogram`, `histc`, `searchsorted`, `quantile`,
    `percentile`, `median`, `mode`, `nanmean`, `nanmedian`, `cov`, `corrcoef`,
    `autocorr`, `entropy`.
15. `src/tensor/factory.rs` — `linspace`, `logspace`, `geomspace`, `arange`,
    `randint`, `randperm`, `multinomial`, `bernoulli`, `diag`, `hankel`, `toeplitz`,
    `vander`, `meshgrid`, `cartesian_prod`, `cross`.
16. `src/tensor/ops_fused.rs` — fused kernels: `bias_add_act`, fused layernorm
    (+residual), `rmsnorm`, `softmax_fused` (online max-tracking), `dropout_add`,
    `gelu_add`, `silu_mul`, `adam_update`, `sgd_update`, fused gradient clipping.
17. `src/tensor/ops_nd.rs` — `apply_along_axis`, `apply_over_axes`, `vectorize`,
    `iter_nd`/`ndenumerate`/`ndindex`, `block_diag`, `stack_grid`.
18. `src/tensor/neural.rs` — `embedding`, `embedding_lookup`, `one_hot`,
    `pad_sequence`, `mask_pad`, `index_put`, `index_add_`, `scatter_add_`,
    `flatten`, `unflatten`, `ravel`.
19. `src/tensor/function.rs` — `map_elementwise`, `zip_map`, `fold_over`,
    `scalar_fn` dispatch table for 100+ scalar math functions.
20. `src/tensor/random_ops.rs` — `normal_`, `uniform_`, `bernoulli_`, `dropout`
    (+`dropout2d/3d` with mask), `shuffle`, `gaussian_noise`, `uniform_noise`.

## B. STRENGTHEN EXISTING MODULES

1. `src/tensor/linalg.rs` — add `lu` + `lu_solve`, `qr_solve`, `svd_solve`, `pinv`
   (via SVD), all p-norms, `condition_number`, `null_space`, `matrix_power`, `eigh`,
   `tridiagonal_solve`, Gauss elimination with partial pivoting, `MatrixDecomposition`
   enum unifying decompositions.
2. `src/tensor/reduction.rs` — add `norm_along_axis`, `sum_squares`, `log_sum_exp`,
   `all`, `any`, `nansum`, `nanvar`, `nanstd`, `ptp`, axis-aware `argreduce`.
3. `src/tensor/impl.rs` — add `into_iter`, `to_vec_2d/3d`, `contiguous()`, `split`,
   `chunk`, `tensor_split`, `expand`, `expand_as`, `squeeze`, `unsqueeze`, `flatten`,
   `unflatten`, `dtype_cast`, `to_device`, `cpu()`.
4. `src/shape.rs` — add `Shape::merge`, `split`, `broadcast_to`, `broadcast_shapes`,
   `permuted`, `transposed`, `is_broadcastable_with`, `narrow`, `expanded`, validation
   suite.
5. `src/memory.rs` — add page-aligned allocations, `MemoryPool` with free-list,
   `MemoryArena` with bump reset, `MemoryTracker` leak detection, `MemoryStats`.
6. `src/random.rs` — add `PCG32`, `SplitMix64`, `ChaCha8` stream, `SeedSeq`/`SeedExt`
   reproducible seeding, `Normal/Gamma/Beta/Exponential/Cauchy/LogNormalDist`,
   `ShuffleSeq`.
7. `src/serialization.rs` — add binary format with magic header + CRC32, multi-tensor
   archives, endianness handling, versioned format with migration.
8. `src/device.rs` — add `DeviceInfo`, thread-local device stack, device predicates,
   policy hooks for `Config`.

## C. HARDENING (apply everywhere)

1. **Numerical stability**: softmax, logsumexp, var/std, pow, erf, gamma must use stable
   formulations; `debug_assert!`-only fast math on fast paths.
2. **Safety**: checked indexing by default; `unsafe`-accelerated kernels only where
   sound, with documented safety invariants; never unsound code.
3. **Errors**: fallible APIs return `BrainResult<T>` using the crate's existing error
   types; no panics in library code except documented internal invariants.
4. **Tests**: every new file ends with `#[cfg(test)] mod tests` with at least 30 tests:
   basic behavior, edge cases (empty, size-1, broadcasts), hand-computed known values,
   round-trip properties (inv∘matmul, ifft∘fft, q∘dq), tolerance checks
   (`atol=1e-9` stable ops, `1e-5` FFT/linalg).
5. **Docs**: `//!` module docs and `///` item docs with `# Example` where feasible.
6. **Perf**: hot loops written to autovectorize; `#[cold]` on error paths,
   `#[inline(always)]` on tiny kernels.

## D. INTEGRATION

1. `src/lib.rs` — export all new modules, re-export key types at crate root, extend
   `prelude`, update `module_info()` and `module_count()`.
2. `src/tensor/mod.rs` — declare submodules, re-export public items, extend `TensorStats`.
3. `Cargo.toml` — keep deps minimal, bump version to `0.2.0`.

## E. QUALITY GATES

1. `cargo check -p brain-core` — zero errors and warnings.
2. `cargo test -p brain-core` — ALL tests pass (existing + new).
3. `cargo clippy -p brain-core -- -D warnings` (if available) — clean.
4. No new external dependencies.
5. Final report: per-file line counts, total lines, capabilities per module.

## Definition of Done

- **API coverage**: at or beyond PyTorch's CPU tensor surface (all A–B ops exist/work).
- **Performance**: matmul/conv/reductions cache-blocked and thread-aware; FFT recursive
  and cache-friendly; hot kernels autovectorize.
- **Robustness**: edge shapes (0-dims, 1-elem, broadcast conflicts) handled cleanly.
- **Size**: brain-core reaches 50+ files, ~150K–300K lines of documented, tested Rust.

Work module by module, keeping the crate compiling and tests green after each module.
Correctness before speed, speed before cosmetics. This prompt covers ONLY `brain-core`
— do not touch other crates. When all gates pass, report the final summary.
