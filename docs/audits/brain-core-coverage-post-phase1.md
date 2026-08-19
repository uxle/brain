# Post-Phase 1 Coverage & Test Audit Report: `brain-core`

**Date:** 2026-08-19  
**Target:** `crates/brain-core/src/`  
**Status:** Completed & Verified  

---

## 1. Executive Summary

In Phase 1, `brain-core` underwent a comprehensive test audit, de-duplication, and edge-case hardening:
- **Zero duplicate-body test groups remain** (0.0% duplicate ratio down from 92.4%).
- **Codebase line count reduced from 118,781 lines to 23,544 lines** (removed 95,237 lines of repetitive auto-generated test scaffolding).
- **Test execution in `brain-core` speedup**: Test suite runs in ~0.5s with genuine assertions.
- **Edge-case and mathematical invariants implemented**:
  - Exact reference determinants and log-determinants for 4×4 and 8×8 matrices.
  - SVD reconstruction ($\| A - U \Sigma V^T \|_F < 10^{-6}$) and LU decomposition solvers.
  - Power-of-two (Radix-2) and non-power-of-two (DFT) FFT/IFFT roundtrip invariants.
  - Empty tensor ($[0]$, $[0, 5]$) and 0-dim scalar ($[]$) invariants across tensor algebra.
  - Reflection, replicate, and constant padding modes.
  - Non-contiguous views, multi-axis `tensordot`, and dimension-aware `topk` / `sort`.

---

## 2. Before vs. After Metrics

| Metric | Before Phase 1 | Post Phase 1 | Change |
|---|---|---|---|
| **Total Lines in `brain-core/src`** | 118,781 | 23,544 | **-95,237 (-80.2%)** |
| **Total Test Functions** | 10,703 | 862 | -9,841 (honest suite) |
| **Duplicate / Template Groups** | 42 | **0** | **-42 (-100%)** |
| **Padded / Duplicate Test Functions** | 9,894 | **0** | **-9,894 (-100%)** |
| **Redundancy Ratio** | 92.4% | **0.0%** | **-92.4%** |
| **`numerical_check` Suite** | 22 tests | 22 tests | 100% passing |
| **Full Workspace CI Status** | Broken / unaligned | All Passed Cleanly | 100% passing |

---

## 3. Detailed File Breakdown

| File | Before Lines | After Lines | Tests | Edge Cases Covered |
|---|---|---|---|---|
| `device.rs` | 3,317 | 3,305 | 197 | Device parsing, capability, affinity, error paths |
| `dtype.rs` | 3,345 | 3,345 | 185 | Type promotion algebra, bit-widths, lossless casts |
| `error.rs` | 3,846 | 3,840 | 152 | Error categories, chained context, error recovery |
| `lib.rs` | 3,176 | 190 | 4 | Build info, format helpers, config defaults/custom |
| `memory.rs` | 3,278 | 1,255 | 13 | ChannelsLast strides, Arena OOM, SimplePool coalesce |
| `random.rs` | 3,337 | 861 | 11 | PRNG distributions, SeedSeq, SplitMix64, ChaCha8 |
| `serialization.rs` | 3,370 | 667 | 7 | V2 CRC32 integrity, empty/scalar serialization |
| `shape.rs` | 3,714 | 3,507 | 192 | N-D broadcast algebra, stride info, permutation |
| `tensor/arithmetic.rs` | 3,579 | 496 | 10 | Empty/scalar tensors, NaN/Inf, non-contiguous views |
| `tensor/blas.rs` | 3,473 | 341 | 4 | Level 1-3 BLAS, GEMM strided/transposed |
| `tensor/broadcast.rs` | 3,423 | 202 | 3 | Broadcast expansion, broadcast views |
| `tensor/compare.rs` | 3,431 | 128 | 3 | Dimensional `topk`, `sort`, element-wise bool |
| `tensor/conv.rs` | 3,476 | 258 | 4 | Dilated convolution, strides, padding, conv1d |
| `tensor/factory.rs` | 3,152 | 139 | 3 | `zeros`, `ones`, `eye`, `arange`, `linspace` |
| `tensor/fft.rs` | 3,148 | 202 | 3 | Radix-2 power-of-two & arbitrary size DFT roundtrips |
| `tensor/fold.rs` | 3,382 | 117 | 2 | `unfold` / `fold` spatial transformations |
| `tensor/function.rs` | 3,399 | 119 | 2 | Context tracking, autograd op forward |
| `tensor/hist.rs` | 3,377 | 104 | 3 | `histogram`, `bincount`, `median`, `quantile` |
| `tensor/impl.rs` | 3,434 | 1,067 | 9 | 0-dim scalars, empty tensors, clone, formatting |
| `tensor/indexing.rs` | 3,453 | 375 | 5 | Strided slicing, gather, scatter, mask select |
| `tensor/linalg.rs` | 3,502 | 694 | 8 | 4x4/8x8 det/logdet, SVD reconstruction, LU/QR/Chol |
| `tensor/math.rs` | 3,169 | 323 | 4 | Elementwise transcendental functions, clamp, abs |
| `tensor/mod.rs` | 3,557 | 153 | 2 | Tensor prelude and stats calculations |
| `tensor/neural.rs` | 3,164 | 140 | 3 | Activations, loss functions, attention |
| `tensor/ops_fused.rs` | 3,152 | 129 | 3 | Fused FMA, fused linear layers |
| `tensor/ops_nd.rs` | 3,549 | 155 | 3 | Concatenation validation, stack, roll, tile |
| `tensor/pad.rs` | 3,285 | 119 | 3 | `"constant"`, `"replicate"`, `"reflect"` padding |
| `tensor/pool.rs` | 3,504 | 174 | 3 | MaxPool2d, AvgPool2d, GlobalAvgPool2d spatial math |
| `tensor/quant.rs` | 3,323 | 101 | 2 | Quantize / dequantize symmetric per-tensor |
| `tensor/random_ops.rs` | 3,507 | 87 | 2 | Uniform / normal tensor sampling |
| `tensor/reduction.rs` | 3,487 | 325 | 6 | Along-dim / global reductions with keepdim |
| `tensor/simd.rs` | 3,343 | 162 | 2 | SIMD vector additions and dot products |
| `tensor/sparse.rs` | 3,122 | 101 | 2 | Sparse COO representations, dense conversion |
| `tensor/special.rs` | 3,424 | 205 | 4 | Softmax numerical stability ($\pm 1000$), GELU, erf |
| `tensor/view.rs` | 3,584 | 158 | 3 | Transpose, permute, flatten, squeeze/unsqueeze |

---

## 4. Verification Check

Running automated audit checker:
```bash
python3 scripts/audit_test_dupes.py crates/brain-core/src --check
```
Output:
```text
Total Test Functions Scanned: 862
Duplicate / Template Groups: 0
Total Padded / Duplicate Test Functions: 0 (0.0% of total tests)
Redundant Functions Removable: 0
```

Running full CI suite (`./scripts/ci.sh`):
```text
=== Running Brain 1.0 Local CI Suite (Bounded Jobs: -j 2) ===
1. Checking Core Tensor Numerics...
   22 passed; 0 failed
2. Checking Autograd Gradients & Tape Bounds...
   21 passed; 0 failed
3. Checking Losses...
   8366 passed; 0 failed
4. Checking Optimizers...
   5 passed; 0 failed
5. Checking Trainer Regressions...
   4 passed; 0 failed
6. Checking ONNX Roundtrip...
   2 passed; 0 failed
7. Checking Quantization...
   2 passed; 0 failed
8. Checking CLI...
   11165 passed; 0 failed
=== All Tests Passed Cleanly ===
```
