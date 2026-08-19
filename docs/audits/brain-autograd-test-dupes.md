# Test Duplication & Inflation Audit Report

- **Target Directory:** `crates/brain-autograd/src`
- **Total Test Functions Scanned:** 13744
- **Duplicate / Template Groups:** 38
- **Total Padded / Duplicate Test Functions:** 13741 (100.0% of total tests)
- **Redundant Functions Removable:** 13703

## Summary by File

| File | Total Tests | Duplicated Tests | Redundancy Ratio |
|---|---|---|---|
| `crates/brain-autograd/src/backward/grad.rs` | 294 | 294 | 100.0% |
| `crates/brain-autograd/src/backward/mod.rs` | 369 | 369 | 100.0% |
| `crates/brain-autograd/src/backward/topo.rs` | 330 | 327 | 99.1% |
| `crates/brain-autograd/src/checkpoint/cpu_offload.rs` | 365 | 365 | 100.0% |
| `crates/brain-autograd/src/checkpoint/mod.rs` | 411 | 411 | 100.0% |
| `crates/brain-autograd/src/checkpoint/offload.rs` | 412 | 412 | 100.0% |
| `crates/brain-autograd/src/checkpoint/selective.rs` | 254 | 254 | 100.0% |
| `crates/brain-autograd/src/engine/mixed.rs` | 202 | 202 | 100.0% |
| `crates/brain-autograd/src/engine/mod.rs` | 554 | 554 | 100.0% |
| `crates/brain-autograd/src/engine/parallel.rs` | 326 | 326 | 100.0% |
| `crates/brain-autograd/src/grad_fns/arith.rs` | 295 | 295 | 100.0% |
| `crates/brain-autograd/src/grad_fns/composite.rs` | 271 | 271 | 100.0% |
| `crates/brain-autograd/src/grad_fns/loss_grad.rs` | 365 | 365 | 100.0% |
| `crates/brain-autograd/src/grad_fns/mod.rs` | 188 | 188 | 100.0% |
| `crates/brain-autograd/src/grad_fns/nnops.rs` | 362 | 362 | 100.0% |
| `crates/brain-autograd/src/graph_closure.rs` | 246 | 246 | 100.0% |
| `crates/brain-autograd/src/lib.rs` | 462 | 462 | 100.0% |
| `crates/brain-autograd/src/ops/activation_grad.rs` | 411 | 411 | 100.0% |
| `crates/brain-autograd/src/ops/binary.rs` | 295 | 295 | 100.0% |
| `crates/brain-autograd/src/ops/broadcast_grad.rs` | 474 | 474 | 100.0% |
| `crates/brain-autograd/src/ops/conv_grad.rs` | 300 | 300 | 100.0% |
| `crates/brain-autograd/src/ops/fft_grad.rs` | 474 | 474 | 100.0% |
| `crates/brain-autograd/src/ops/index_grad.rs` | 473 | 473 | 100.0% |
| `crates/brain-autograd/src/ops/linalg_grad.rs` | 413 | 413 | 100.0% |
| `crates/brain-autograd/src/ops/mod.rs` | 474 | 474 | 100.0% |
| `crates/brain-autograd/src/ops/pool_grad.rs` | 473 | 473 | 100.0% |
| `crates/brain-autograd/src/ops/quant_grad.rs` | 414 | 414 | 100.0% |
| `crates/brain-autograd/src/ops/reduction_grad.rs` | 473 | 473 | 100.0% |
| `crates/brain-autograd/src/ops/sparse_grad.rs` | 414 | 414 | 100.0% |
| `crates/brain-autograd/src/ops/tensor_grad.rs` | 472 | 472 | 100.0% |
| `crates/brain-autograd/src/ops/unary.rs` | 318 | 318 | 100.0% |
| `crates/brain-autograd/src/tape/builder.rs` | 413 | 413 | 100.0% |
| `crates/brain-autograd/src/tape/fused.rs` | 367 | 367 | 100.0% |
| `crates/brain-autograd/src/tape/mod.rs` | 465 | 465 | 100.0% |
| `crates/brain-autograd/src/tape/node.rs` | 413 | 413 | 100.0% |
| `crates/brain-autograd/src/tape/prune.rs` | 367 | 367 | 100.0% |
| `crates/brain-autograd/src/value.rs` | 135 | 135 | 100.0% |

## Top Duplicate Groups

### Group 1: 554 identical functions (e.g. `test_engine_mod_stress_001` in `crates/brain-autograd/src/engine/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/engine/mod.rs:20`):
```rust
fn test_engine_mod_stress_001() {
        let scaler = GradScaler::default();
        assert!(scaler.scale_factor() > 0.0);
    }
```

### Group 2: 474 identical functions (e.g. `test_ops_mod_stress_001` in `crates/brain-autograd/src/ops/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/ops/mod.rs:31`):
```rust
fn test_ops_mod_stress_001() {
        let a = Value::scalar(1.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }
```

### Group 3: 474 identical functions (e.g. `test_broadcast_grad_stress_001` in `crates/brain-autograd/src/ops/broadcast_grad.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/ops/broadcast_grad.rs:29`):
```rust
fn test_broadcast_grad_stress_001() {
        let g = Tensor::from_slice(&[1.1, 2.0, 3.0, 4.0], vec![2, 2]);
        let unb = unbroadcast(&g, &[2, 1]).unwrap();
        assert_eq!(unb.shape(), &[2, 1]);
    }
```

### Group 4: 474 identical functions (e.g. `test_fft_grad_stress_001` in `crates/brain-autograd/src/ops/fft_grad.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/ops/fft_grad.rs:24`):
```rust
fn test_fft_grad_stress_001() {
        let g = Tensor::from_slice(&[1.1, 0.0], vec![2]);
        let g_in = grad_fft1d(&g).unwrap();
        assert_eq!(g_in.shape(), &[2]);
    }
```

### Group 5: 473 identical functions (e.g. `test_reduction_grad_stress_001` in `crates/brain-autograd/src/ops/reduction_grad.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/ops/reduction_grad.rs:36`):
```rust
fn test_reduction_grad_stress_001() {
        let g = Tensor::scalar(2.1);
        let unscaled = grad_sum_axis(&g, &[2, 2], 0).unwrap();
        assert_eq!(unscaled.shape(), &[2, 2]);
    }
```

### Group 6: 473 identical functions (e.g. `test_index_grad_stress_001` in `crates/brain-autograd/src/ops/index_grad.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/ops/index_grad.rs:35`):
```rust
fn test_index_grad_stress_001() {
        let g = Tensor::from_slice(&[1.1, 2.0], vec![1, 2]);
        let gw = grad_embedding(&g, 4, 2, &[1]).unwrap();
        assert_eq!(gw.shape(), &[4, 2]);
    }
```

### Group 7: 473 identical functions (e.g. `test_pool_grad_stress_001` in `crates/brain-autograd/src/ops/pool_grad.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/ops/pool_grad.rs:231`):
```rust
fn test_pool_grad_stress_001() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }
```

### Group 8: 472 identical functions (e.g. `test_tensor_view_grad_stress_001` in `crates/brain-autograd/src/ops/tensor_grad.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/ops/tensor_grad.rs:39`):
```rust
fn test_tensor_view_grad_stress_001() {
        let g = Tensor::from_slice(&[1.1, 2.0, 3.0, 4.0], vec![4]);
        let reshaped = grad_reshape(&g, &[2, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[2, 2]);
    }
```

### Group 9: 465 identical functions (e.g. `test_tape_lifecycle_stress_001` in `crates/brain-autograd/src/tape/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/tape/mod.rs:100`):
```rust
fn test_tape_lifecycle_stress_001() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![1], vec![2], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }
```

### Group 10: 462 identical functions (e.g. `test_autograd_lib_stress_001` in `crates/brain-autograd/src/lib.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/lib.rs:108`):
```rust
fn test_autograd_lib_stress_001() {
        let x = Value::scalar(1.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }
```

### Group 11: 414 identical functions (e.g. `test_sparse_grad_stress_001` in `crates/brain-autograd/src/ops/sparse_grad.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/ops/sparse_grad.rs:29`):
```rust
fn test_sparse_grad_stress_001() {
        let b = Tensor::from_slice(&[1.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }
```

### Group 12: 413 identical functions (e.g. `test_linalg_grad_stress_001` in `crates/brain-autograd/src/ops/linalg_grad.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/ops/linalg_grad.rs:36`):
```rust
fn test_linalg_grad_stress_001() {
        let a_inv = Tensor::from_slice(&[1.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }
```

### Group 13: 413 identical functions (e.g. `test_tape_builder_stress_001` in `crates/brain-autograd/src/tape/builder.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/tape/builder.rs:43`):
```rust
fn test_tape_builder_stress_001() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![1], vec![2], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }
```

### Group 14: 413 identical functions (e.g. `test_tape_op_record_stress_001` in `crates/brain-autograd/src/tape/node.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/tape/node.rs:42`):
```rust
fn test_tape_op_record_stress_001() {
        let rec = OpRecord::new("add", vec![1, 2], vec![3], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }
```

### Group 15: 412 identical functions (e.g. `test_recompute_graph_stress_001` in `crates/brain-autograd/src/checkpoint/offload.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/checkpoint/offload.rs:47`):
```rust
fn test_recompute_graph_stress_001() {
        let t = Arc::new(Tensor::scalar(1.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }
```

### Group 16: 411 identical functions (e.g. `test_checkpoint_budget_stress_001` in `crates/brain-autograd/src/checkpoint/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/checkpoint/mod.rs:53`):
```rust
fn test_checkpoint_budget_stress_001() {
        let b = BudgetCheckpoint::new(11, 1024 * 1024);
        let mask = b.compute_checkpoint_mask();
        assert_eq!(mask.len(), 11);
        assert!(mask[0]);
    }
```

### Group 17: 411 identical functions (e.g. `test_activation_grad_stress_001` in `crates/brain-autograd/src/ops/activation_grad.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/ops/activation_grad.rs:58`):
```rust
fn test_activation_grad_stress_001() {
        let x = Tensor::scalar(0.55);
        let g = Tensor::scalar(1.0);
        let dg = grad_gelu(&x, &g).unwrap();
        assert!(dg.get(0) > 0.0);
    }
```

### Group 18: 369 identical functions (e.g. `test_backward_mod_stress_001` in `crates/brain-autograd/src/backward/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/backward/mod.rs:18`):
```rust
fn test_backward_mod_stress_001() {
        let mut a = Value::scalar(1.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }
```

### Group 19: 367 identical functions (e.g. `test_tape_fusion_stress_001` in `crates/brain-autograd/src/tape/fused.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/tape/fused.rs:39`):
```rust
fn test_tape_fusion_stress_001() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![1], vec![2], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }
```

### Group 20: 367 identical functions (e.g. `test_tape_pruner_stress_001` in `crates/brain-autograd/src/tape/prune.rs`)
- Files involved: 1
- Sample definition (`crates/brain-autograd/src/tape/prune.rs:39`):
```rust
fn test_tape_pruner_stress_001() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![1], vec![2], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[2]);
        assert_eq!(out.op_count(), 1);
    }
```
