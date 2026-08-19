# Test Duplication & Inflation Audit Report

- **Target Directory:** `crates/brain-nn/src`
- **Total Test Functions Scanned:** 12430
- **Duplicate / Template Groups:** 34
- **Total Padded / Duplicate Test Functions:** 12417 (99.9% of total tests)
- **Redundant Functions Removable:** 12383

## Summary by File

| File | Total Tests | Duplicated Tests | Redundancy Ratio |
|---|---|---|---|
| `crates/brain-nn/src/activations/gelu.rs` | 411 | 411 | 100.0% |
| `crates/brain-nn/src/activations/mod.rs` | 473 | 473 | 100.0% |
| `crates/brain-nn/src/activations/relu.rs` | 329 | 329 | 100.0% |
| `crates/brain-nn/src/activations/sigmoid.rs` | 471 | 471 | 100.0% |
| `crates/brain-nn/src/activations/softmax.rs` | 298 | 298 | 100.0% |
| `crates/brain-nn/src/activations/swish.rs` | 363 | 363 | 100.0% |
| `crates/brain-nn/src/containers/mod.rs` | 555 | 555 | 100.0% |
| `crates/brain-nn/src/containers/seq.rs` | 193 | 193 | 100.0% |
| `crates/brain-nn/src/containers/sequential2.rs` | 183 | 183 | 100.0% |
| `crates/brain-nn/src/dropout/alpha.rs` | 329 | 329 | 100.0% |
| `crates/brain-nn/src/dropout/dropout.rs` | 271 | 271 | 100.0% |
| `crates/brain-nn/src/dropout/mod.rs` | 555 | 555 | 100.0% |
| `crates/brain-nn/src/hooks.rs` | 366 | 366 | 100.0% |
| `crates/brain-nn/src/init/kaiming.rs` | 217 | 217 | 100.0% |
| `crates/brain-nn/src/init/mod.rs` | 473 | 473 | 100.0% |
| `crates/brain-nn/src/init/schedule.rs` | 331 | 331 | 100.0% |
| `crates/brain-nn/src/init/uniform.rs` | 274 | 274 | 100.0% |
| `crates/brain-nn/src/layers/activation_layers.rs` | 414 | 414 | 100.0% |
| `crates/brain-nn/src/layers/attention.rs` | 367 | 367 | 100.0% |
| `crates/brain-nn/src/layers/conv.rs` | 408 | 407 | 99.8% |
| `crates/brain-nn/src/layers/conv2d.rs` | 2 | 0 | 0.0% |
| `crates/brain-nn/src/layers/conv_transpose.rs` | 412 | 411 | 99.8% |
| `crates/brain-nn/src/layers/embedding.rs` | 328 | 328 | 100.0% |
| `crates/brain-nn/src/layers/linear.rs` | 363 | 363 | 100.0% |
| `crates/brain-nn/src/layers/linear2d.rs` | 299 | 299 | 100.0% |
| `crates/brain-nn/src/layers/mod.rs` | 414 | 414 | 100.0% |
| `crates/brain-nn/src/layers/multihead.rs` | 363 | 363 | 100.0% |
| `crates/brain-nn/src/layers/norm.rs` | 416 | 416 | 100.0% |
| `crates/brain-nn/src/layers/pool.rs` | 3 | 0 | 0.0% |
| `crates/brain-nn/src/layers/recurrent.rs` | 2 | 0 | 0.0% |
| `crates/brain-nn/src/layers/rnn_cells.rs` | 2 | 0 | 0.0% |
| `crates/brain-nn/src/module/mod.rs` | 217 | 217 | 100.0% |
| `crates/brain-nn/src/module/parameter.rs` | 299 | 299 | 100.0% |
| `crates/brain-nn/src/normalization/batch.rs` | 2 | 0 | 0.0% |
| `crates/brain-nn/src/normalization/group.rs` | 472 | 472 | 100.0% |
| `crates/brain-nn/src/normalization/layer.rs` | 364 | 364 | 100.0% |
| `crates/brain-nn/src/normalization/mod.rs` | 414 | 414 | 100.0% |
| `crates/brain-nn/src/normalization/rms.rs` | 364 | 364 | 100.0% |
| `crates/brain-nn/src/pruning.rs` | 413 | 413 | 100.0% |

## Top Duplicate Groups

### Group 1: 555 identical functions (e.g. `test_container_mod_stress_001` in `crates/brain-nn/src/containers/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/containers/mod.rs:19`):
```rust
fn test_container_mod_stress_001() {
        let seq = Sequential::new();
        assert_eq!(seq.len(), 0);
    }
```

### Group 2: 555 identical functions (e.g. `test_dropout_mod_stress_001` in `crates/brain-nn/src/dropout/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/dropout/mod.rs:20`):
```rust
fn test_dropout_mod_stress_001() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }
```

### Group 3: 473 identical functions (e.g. `test_act_mod_stress_001` in `crates/brain-nn/src/activations/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/activations/mod.rs:33`):
```rust
fn test_act_mod_stress_001() {
        let t = Tensor::from_vec(vec![-1.0, 1.0], vec![2]);
        let r = relu(&t);
        assert_eq!(r.to_vec(), vec![0.0, 1.0]);
    }
```

### Group 4: 473 identical functions (e.g. `test_init_mod_stress_001` in `crates/brain-nn/src/init/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/init/mod.rs:34`):
```rust
fn test_init_mod_stress_001() {
        let (fan_in, fan_out) = calculate_fan(&[64, 32, 3, 3]);
        assert_eq!(fan_in, 32 * 9);
        assert_eq!(fan_out, 64 * 9);
    }
```

### Group 5: 472 identical functions (e.g. `test_groupnorm_stress_001` in `crates/brain-nn/src/normalization/group.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/normalization/group.rs:44`):
```rust
fn test_groupnorm_stress_001() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }
```

### Group 6: 471 identical functions (e.g. `test_sigmoid_stress_001` in `crates/brain-nn/src/activations/sigmoid.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/activations/sigmoid.rs:47`):
```rust
fn test_sigmoid_stress_001() {
        let t = Tensor::from_vec(vec![0.0], vec![1]);
        assert!((sigmoid(&t).to_vec()[0] - 0.5).abs() < 1e-9);
        assert!((tanh(&t).to_vec()[0] - 0.0).abs() < 1e-9);
    }
```

### Group 7: 416 identical functions (e.g. `test_layers_norm_stress_001` in `crates/brain-nn/src/layers/norm.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/layers/norm.rs:17`):
```rust
fn test_layers_norm_stress_001() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }
```

### Group 8: 414 identical functions (e.g. `test_layers_mod_stress_001` in `crates/brain-nn/src/layers/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/layers/mod.rs:37`):
```rust
fn test_layers_mod_stress_001() {
        let ident = Identity;
        let t = Tensor::zeros(vec![2, 2]);
        let out = ident.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
    }
```

### Group 9: 414 identical functions (e.g. `test_act_layers_stress_001` in `crates/brain-nn/src/layers/activation_layers.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/layers/activation_layers.rs:36`):
```rust
fn test_act_layers_stress_001() {
        let r = ReLU;
        let t = Tensor::from_vec(vec![-1.0, 2.0], vec![2]);
        let out = r.forward(&t).unwrap();
        assert_eq!(out.to_vec(), vec![0.0, 2.0]);
    }
```

### Group 10: 414 identical functions (e.g. `test_norm_mod_stress_001` in `crates/brain-nn/src/normalization/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/normalization/mod.rs:31`):
```rust
fn test_norm_mod_stress_001() {
        let ln = LayerNorm::new(vec![2], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 3.0], vec![1, 2]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 2]);
    }
```

### Group 11: 413 identical functions (e.g. `test_pruning_stress_001` in `crates/brain-nn/src/pruning.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/pruning.rs:41`):
```rust
fn test_pruning_stress_001() {
        let w = Tensor::from_vec(vec![0.1, 0.5, 0.9, 0.2], vec![4]);
        let mask = PruningMask::from_magnitude(&w, 0.5);
        let pruned = mask.apply(&w);
        assert_eq!(pruned.shape(), &[4]);
    }
```

### Group 12: 411 identical functions (e.g. `test_gelu_stress_001` in `crates/brain-nn/src/activations/gelu.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/activations/gelu.rs:55`):
```rust
fn test_gelu_stress_001() {
        let t = Tensor::from_vec(vec![0.0, 1.0], vec![2]);
        let g = gelu(&t);
        assert!((g.to_vec()[0] - 0.0).abs() < 1e-9);
        assert!(g.to_vec()[1] > 0.8);
    }
```

### Group 13: 411 identical functions (e.g. `test_conv_transpose_stress_001` in `crates/brain-nn/src/layers/conv_transpose.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/layers/conv_transpose.rs:175`):
```rust
fn test_conv_transpose_stress_001() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }
```

### Group 14: 407 identical functions (e.g. `test_conv_stress_001` in `crates/brain-nn/src/layers/conv.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/layers/conv.rs:107`):
```rust
fn test_conv_stress_001() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }
```

### Group 15: 367 identical functions (e.g. `test_attention_stress_001` in `crates/brain-nn/src/layers/attention.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/layers/attention.rs:105`):
```rust
fn test_attention_stress_001() {
        let q = Tensor::zeros(vec![1, 4, 8]);
        let k = Tensor::zeros(vec![1, 4, 8]);
        let v = Tensor::zeros(vec![1, 4, 8]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[1, 4, 8]);
    }
```

### Group 16: 366 identical functions (e.g. `test_hooks_stress_001` in `crates/brain-nn/src/hooks.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/hooks.rs:48`):
```rust
fn test_hooks_stress_001() {
        let mut reg = HookRegistry::new();
        reg.register_pre_hook(|x| x.clone());
        reg.register_post_hook(|_inp, out| out.clone());
        assert_eq!(reg.pre_hooks.len(), 1);
        assert_eq!(reg.post_hooks.len(), 1);
    }
```

### Group 17: 364 identical functions (e.g. `test_layernorm_stress_001` in `crates/brain-nn/src/normalization/layer.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/normalization/layer.rs:74`):
```rust
fn test_layernorm_stress_001() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }
```

### Group 18: 364 identical functions (e.g. `test_rmsnorm_stress_001` in `crates/brain-nn/src/normalization/rms.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/normalization/rms.rs:73`):
```rust
fn test_rmsnorm_stress_001() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }
```

### Group 19: 363 identical functions (e.g. `test_swish_stress_001` in `crates/brain-nn/src/activations/swish.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/activations/swish.rs:75`):
```rust
fn test_swish_stress_001() {
        let t = Tensor::from_vec(vec![0.0, 1.0], vec![2]);
        let s = silu(&t);
        let m = mish(&t);
        assert!((s.to_vec()[0] - 0.0).abs() < 1e-9);
        assert!((m.to_vec()[0] - 0.0).abs() < 1e-9);
    }
```

### Group 20: 363 identical functions (e.g. `test_linear_stress_001` in `crates/brain-nn/src/layers/linear.rs`)
- Files involved: 1
- Sample definition (`crates/brain-nn/src/layers/linear.rs:91`):
```rust
fn test_linear_stress_001() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }
```
