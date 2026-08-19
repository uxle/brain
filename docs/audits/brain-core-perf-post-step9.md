# Post-Step 9 Performance & Multi-Threading Report: `brain-core`

**Date:** 2026-08-19  
**Target:** `crates/brain-core/` (CPU Acceleration & Parallelization)  
**Status:** Completed & Verified  

---

## 1. Executive Summary

In Step 9 of the framework architecture, `brain-core` received high-performance CPU parallelization, cache-blocked GEMM, and vector unrolling using pure, zero-dependency standard library multi-threading (`std::thread::scope`):

1. **Multi-Threaded 2D GEMM (`matmul`)**:
   - Implemented row-chunked parallel execution across available CPU cores (`std::thread::available_parallelism()`).
   - Combined with $64 \times 64$ L1/L2 cache-tiling and 4-way loop unrolling for inner dot products.
   - Preserves exact numerical floating-point equivalence while maximizing CPU throughput.
2. **Multi-Threaded Batched GEMM**:
   - Parallelized across batch dimensions for 3D/4D tensors using scoped threads.
3. **Multi-Threaded Elementwise Mappings (`map`, `map2`)**:
   - Added chunked multi-threaded evaluation for tensors with $\ge 8,192$ elements.
4. **Contiguous Memory Slicing**:
   - `cat`, `outer`, and `kron` operations now execute via direct contiguous slice memory copies (`extend_from_slice` / `memcpy`).
5. **Integration & Numerical Correctness Verification**:
   - Added large matrix tests ($128 \times 128$ identity/dense GEMM, $4 \times 32 \times 32$ batched GEMM, and $16,384$-element parallel map) in [`crates/brain-core/tests/numerical_check.rs`](crates/brain-core/tests/numerical_check.rs).
   - 100% CI pass rate across all 33 crates (`./scripts/ci.sh`).

---

## 2. Test Execution

```bash
cargo test -p brain-core --test numerical_check -j 2
```
```text
running 25 tests
test check_arange ... ok
test check_avg_pool2d_no_pad ... ok
test check_avg_pool2d_with_pad ... ok
test check_avg_pool_counts_valid_only ... ok
test check_conv2d_kernel_larger_than_input_does_not_crash ... ok
test check_conv2d_output_size ... ok
test check_broadcast_map2 ... ok
test check_max_pool2d ... ok
test check_dilated_conv_output ... ok
test check_pad_reflect ... ok
test check_permute ... ok
test check_pool_kernel_larger_than_input_does_not_crash ... ok
test check_dtype_is_lossless_cast ... ok
test check_reduce_along_dim_max ... ok
test check_batched_matmul_broadcast ... ok
test check_reduce_along_dim_sum ... ok
test check_reshape ... ok
test check_transpose_roundtrip ... ok
test check_softmax_correctness ... ok
test check_tensordot_multi_axis ... ok
test check_topk_multi_dim ... ok
test check_global_avg_pool2d ... ok
test check_parallel_batched_gemm ... ok
test check_parallel_elementwise_map ... ok
test check_parallel_gemm_large_matrices ... ok

test result: ok. 25 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```
