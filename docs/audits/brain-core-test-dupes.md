# Test Duplication & Inflation Audit Report

- **Target Directory:** `crates/brain-core/src`
- **Total Test Functions Scanned:** 10703
- **Duplicate / Template Groups:** 42
- **Total Padded / Duplicate Test Functions:** 9894 (92.4% of total tests)
- **Redundant Functions Removable:** 9852

## Summary by File

| File | Total Tests | Duplicated Tests | Redundancy Ratio |
|---|---|---|---|
| `crates/brain-core/src/device.rs` | 199 | 4 | 2.0% |
| `crates/brain-core/src/dtype.rs` | 185 | 0 | 0.0% |
| `crates/brain-core/src/error.rs` | 153 | 2 | 1.3% |
| `crates/brain-core/src/lib.rs` | 380 | 379 | 99.7% |
| `crates/brain-core/src/memory.rs` | 100 | 89 | 89.0% |
| `crates/brain-core/src/random.rs` | 84 | 74 | 88.1% |
| `crates/brain-core/src/serialization.rs` | 149 | 144 | 96.6% |
| `crates/brain-core/src/shape.rs` | 213 | 29 | 13.6% |
| `crates/brain-core/src/tensor/arithmetic.rs` | 215 | 209 | 97.2% |
| `crates/brain-core/src/tensor/blas.rs` | 352 | 349 | 99.1% |
| `crates/brain-core/src/tensor/broadcast.rs` | 361 | 359 | 99.4% |
| `crates/brain-core/src/tensor/compare.rs` | 371 | 369 | 99.5% |
| `crates/brain-core/src/tensor/conv.rs` | 362 | 359 | 99.2% |
| `crates/brain-core/src/tensor/factory.rs` | 381 | 379 | 99.5% |
| `crates/brain-core/src/tensor/fft.rs` | 380 | 379 | 99.7% |
| `crates/brain-core/src/tensor/fold.rs` | 410 | 409 | 99.8% |
| `crates/brain-core/src/tensor/function.rs` | 300 | 299 | 99.7% |
| `crates/brain-core/src/tensor/hist.rs` | 471 | 469 | 99.6% |
| `crates/brain-core/src/tensor/impl.rs` | 167 | 159 | 95.2% |
| `crates/brain-core/src/tensor/indexing.rs` | 313 | 309 | 98.7% |
| `crates/brain-core/src/tensor/linalg.rs` | 268 | 264 | 98.5% |
| `crates/brain-core/src/tensor/math.rs` | 207 | 204 | 98.6% |
| `crates/brain-core/src/tensor/mod.rs` | 380 | 379 | 99.7% |
| `crates/brain-core/src/tensor/neural.rs` | 381 | 379 | 99.5% |
| `crates/brain-core/src/tensor/ops_fused.rs` | 381 | 379 | 99.5% |
| `crates/brain-core/src/tensor/ops_nd.rs` | 381 | 379 | 99.5% |
| `crates/brain-core/src/tensor/pad.rs` | 291 | 289 | 99.3% |
| `crates/brain-core/src/tensor/pool.rs` | 421 | 419 | 99.5% |
| `crates/brain-core/src/tensor/quant.rs` | 360 | 359 | 99.7% |
| `crates/brain-core/src/tensor/random_ops.rs` | 430 | 429 | 99.8% |
| `crates/brain-core/src/tensor/reduction.rs` | 249 | 244 | 98.0% |
| `crates/brain-core/src/tensor/simd.rs` | 320 | 319 | 99.7% |
| `crates/brain-core/src/tensor/sparse.rs` | 380 | 379 | 99.7% |
| `crates/brain-core/src/tensor/special.rs` | 362 | 359 | 99.2% |
| `crates/brain-core/src/tensor/view.rs` | 346 | 344 | 99.4% |

## Top Duplicate Groups

### Group 1: 469 identical functions (e.g. `test_hist_stress_case_001` in `crates/brain-core/src/tensor/hist.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/hist.rs:96`):
```rust
fn test_hist_stress_case_001() {
        let t = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let m = median(&t);
        assert!(m >= 1.0 && m <= 2.0);
    }
```

### Group 2: 429 identical functions (e.g. `test_random_ops_stress_case_001` in `crates/brain-core/src/tensor/random_ops.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/random_ops.rs:77`):
```rust
fn test_random_ops_stress_case_001() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }
```

### Group 3: 419 identical functions (e.g. `test_pool_stress_case_001` in `crates/brain-core/src/tensor/pool.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/pool.rs:154`):
```rust
fn test_pool_stress_case_001() {
        let input = Tensor::full(vec![1, 1, 4, 4], 1.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 1.0);
    }
```

### Group 4: 409 identical functions (e.g. `test_fold_stress_case_001` in `crates/brain-core/src/tensor/fold.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/fold.rs:112`):
```rust
fn test_fold_stress_case_001() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let u = unfold(&t, 0, 2, 1);
        assert_eq!(u.shape(), &[9, 2]);
        assert_eq!(u.get_2d(0, 0), 0.0);
    }
```

### Group 5: 379 identical functions (e.g. `test_lib_core_stress_case_001` in `crates/brain-core/src/lib.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/lib.rs:146`):
```rust
fn test_lib_core_stress_case_001() {
        let cfg = Config::default();
        assert_eq!(cfg.default_dtype, DType::F64);
        let s = format_shape(&[1, 2]);
        assert!(!s.is_empty());
    }
```

### Group 6: 379 identical functions (e.g. `test_tmod_stress_case_001` in `crates/brain-core/src/tensor/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/mod.rs:148`):
```rust
fn test_tmod_stress_case_001() {
        let t = Tensor::full(vec![4], 1.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 1.0);
        assert!(s.is_finite);
    }
```

### Group 7: 379 identical functions (e.g. `test_fft_stress_case_001` in `crates/brain-core/src/tensor/fft.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/fft.rs:117`):
```rust
fn test_fft_stress_case_001() {
        let mut real = vec![1.0, 0.0, 0.0, 0.0];
        let mut imag = vec![0.0; 4];
        fft_radix2(&mut real, &mut imag, false);
        assert_eq!(real[0], 1.0);
    }
```

### Group 8: 379 identical functions (e.g. `test_sparse_stress_case_001` in `crates/brain-core/src/tensor/sparse.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/sparse.rs:92`):
```rust
fn test_sparse_stress_case_001() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 1.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 1.0);
    }
```

### Group 9: 379 identical functions (e.g. `test_factory_stress_case_001` in `crates/brain-core/src/tensor/factory.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/factory.rs:122`):
```rust
fn test_factory_stress_case_001() {
        let v = Tensor::from_slice(&[1.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 1.0);
    }
```

### Group 10: 379 identical functions (e.g. `test_fused_stress_case_001` in `crates/brain-core/src/tensor/ops_fused.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/ops_fused.rs:122`):
```rust
fn test_fused_stress_case_001() {
        let x = Tensor::full(vec![1, 2], 1.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 1.0);
    }
```

### Group 11: 379 identical functions (e.g. `test_nd_stress_case_001` in `crates/brain-core/src/tensor/ops_nd.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/ops_nd.rs:140`):
```rust
fn test_nd_stress_case_001() {
        let t = Tensor::from_slice(&[1.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 1.0);
        assert_eq!(c.get(1), 1.0);
    }
```

### Group 12: 379 identical functions (e.g. `test_neural_stress_case_001` in `crates/brain-core/src/tensor/neural.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/neural.rs:134`):
```rust
fn test_neural_stress_case_001() {
        let p = Tensor::from_slice(&[1.0], vec![1]);
        let t = Tensor::from_slice(&[2.0], vec![1]);
        assert_eq!(mse_loss(&p, &t), 1.0);
        assert_eq!(l1_loss(&p, &t), 1.0);
    }
```

### Group 13: 369 identical functions (e.g. `test_compare_stress_case_001` in `crates/brain-core/src/tensor/compare.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/compare.rs:112`):
```rust
fn test_compare_stress_case_001() {
        let a = Tensor::from_slice(&[1.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }
```

### Group 14: 359 identical functions (e.g. `test_broadcast_stress_case_001` in `crates/brain-core/src/tensor/broadcast.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/broadcast.rs:194`):
```rust
fn test_broadcast_stress_case_001() {
        let t = Tensor::full(vec![1, 2], 1.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 1.0);
        assert_eq!(b.get(5), 1.0);
    }
```

### Group 15: 359 identical functions (e.g. `test_special_stress_case_001` in `crates/brain-core/src/tensor/special.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/special.rs:195`):
```rust
fn test_special_stress_case_001() {
        let x = (1 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }
```

### Group 16: 359 identical functions (e.g. `test_conv_stress_case_001` in `crates/brain-core/src/tensor/conv.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/conv.rs:247`):
```rust
fn test_conv_stress_case_001() {
        let input = Tensor::full(vec![1, 1, 3, 3], 1.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (1 as f64) * 4.0);
    }
```

### Group 17: 359 identical functions (e.g. `test_quant_stress_case_001` in `crates/brain-core/src/tensor/quant.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/quant.rs:94`):
```rust
fn test_quant_stress_case_001() {
        let val = ((1 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }
```

### Group 18: 349 identical functions (e.g. `test_blas_stress_case_001` in `crates/brain-core/src/tensor/blas.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/blas.rs:334`):
```rust
fn test_blas_stress_case_001() {
        let mut y = vec![0.0, 0.0];
        let x = vec![1.0, 1.0];
        axpy(2, 0.5, &x, 1, &mut y, 1);
        assert_eq!(y[0], (1 as f64) * 0.5);
        assert_eq!(y[1], 0.5);
    }
```

### Group 19: 344 identical functions (e.g. `test_view_stress_case_001` in `crates/brain-core/src/tensor/view.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/view.rs:146`):
```rust
fn test_view_stress_case_001() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }
```

### Group 20: 319 identical functions (e.g. `test_simd_stress_case_001` in `crates/brain-core/src/tensor/simd.rs`)
- Files involved: 1
- Sample definition (`crates/brain-core/src/tensor/simd.rs:155`):
```rust
fn test_simd_stress_case_001() {
        let a = vec![1.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 1.0 + 2.0);
        assert_eq!(simd_sum(&a), 1.0 + 1.0);
    }
```
