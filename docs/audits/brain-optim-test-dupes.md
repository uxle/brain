# Test Duplication & Inflation Audit Report

- **Target Directory:** `crates/brain-optim/src`
- **Total Test Functions Scanned:** 8739
- **Duplicate / Template Groups:** 33
- **Total Padded / Duplicate Test Functions:** 8739 (100.0% of total tests)
- **Redundant Functions Removable:** 8706

## Summary by File

| File | Total Tests | Duplicated Tests | Redundancy Ratio |
|---|---|---|---|
| `crates/brain-optim/src/adadelta.rs` | 262 | 262 | 100.0% |
| `crates/brain-optim/src/adagrad.rs` | 263 | 263 | 100.0% |
| `crates/brain-optim/src/adam/mod.rs` | 258 | 258 | 100.0% |
| `crates/brain-optim/src/adam/variants.rs` | 248 | 248 | 100.0% |
| `crates/brain-optim/src/amp.rs` | 231 | 231 | 100.0% |
| `crates/brain-optim/src/builder.rs` | 211 | 211 | 100.0% |
| `crates/brain-optim/src/clipping/adaptive.rs` | 271 | 271 | 100.0% |
| `crates/brain-optim/src/clipping/mod.rs` | 325 | 325 | 100.0% |
| `crates/brain-optim/src/clipping/norm.rs` | 270 | 270 | 100.0% |
| `crates/brain-optim/src/ema.rs` | 250 | 250 | 100.0% |
| `crates/brain-optim/src/lamb.rs` | 260 | 260 | 100.0% |
| `crates/brain-optim/src/lib.rs` | 467 | 467 | 100.0% |
| `crates/brain-optim/src/lion.rs` | 263 | 263 | 100.0% |
| `crates/brain-optim/src/lookahead.rs` | 251 | 251 | 100.0% |
| `crates/brain-optim/src/loss_landscape.rs` | 363 | 363 | 100.0% |
| `crates/brain-optim/src/lr_finder/mod.rs` | 159 | 159 | 100.0% |
| `crates/brain-optim/src/novograd.rs` | 261 | 261 | 100.0% |
| `crates/brain-optim/src/optimizer/mod.rs` | 153 | 153 | 100.0% |
| `crates/brain-optim/src/optimizer/param_group.rs` | 187 | 187 | 100.0% |
| `crates/brain-optim/src/radam.rs` | 261 | 261 | 100.0% |
| `crates/brain-optim/src/rmsprop.rs` | 259 | 259 | 100.0% |
| `crates/brain-optim/src/sam.rs` | 203 | 203 | 100.0% |
| `crates/brain-optim/src/schedulers/cosine.rs` | 287 | 287 | 100.0% |
| `crates/brain-optim/src/schedulers/cyclic.rs` | 457 | 457 | 100.0% |
| `crates/brain-optim/src/schedulers/mod.rs` | 322 | 322 | 100.0% |
| `crates/brain-optim/src/schedulers/onecycle.rs` | 212 | 212 | 100.0% |
| `crates/brain-optim/src/schedulers/plateau.rs` | 396 | 396 | 100.0% |
| `crates/brain-optim/src/schedulers/step.rs` | 277 | 277 | 100.0% |
| `crates/brain-optim/src/schedulers/warmup.rs` | 239 | 239 | 100.0% |
| `crates/brain-optim/src/sgd/mod.rs` | 156 | 156 | 100.0% |
| `crates/brain-optim/src/sgd/nesterov.rs` | 265 | 265 | 100.0% |
| `crates/brain-optim/src/state.rs` | 202 | 202 | 100.0% |
| `crates/brain-optim/src/swa/mod.rs` | 250 | 250 | 100.0% |

## Top Duplicate Groups

### Group 1: 467 identical functions (e.g. `test_lib_root_stress_001` in `crates/brain-optim/src/lib.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/lib.rs:80`):
```rust
fn test_lib_root_stress_001() {
        assert_eq!(VERSION, "0.2.0");
        let b = OptimizerBuilder::new().adam().lr(0.001);
        assert_eq!(b.lr, 0.001);
    }
```

### Group 2: 457 identical functions (e.g. `test_cyclic_schedulers_stress_001` in `crates/brain-optim/src/schedulers/cyclic.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/schedulers/cyclic.rs:150`):
```rust
fn test_cyclic_schedulers_stress_001() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }
```

### Group 3: 396 identical functions (e.g. `test_plateau_schedulers_stress_001` in `crates/brain-optim/src/schedulers/plateau.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/schedulers/plateau.rs:177`):
```rust
fn test_plateau_schedulers_stress_001() {
        let mut plat = ReduceLROnPlateau::new(vec![0.01], PlateauConfig::default());
        assert_eq!(plat.config.patience, 10);
        assert_eq!(plat.config.factor, 0.1);
        assert_eq!(plat.num_bad_epochs, 0);
    }
```

### Group 4: 363 identical functions (e.g. `test_loss_landscape_stress_001` in `crates/brain-optim/src/loss_landscape.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/loss_landscape.rs:78`):
```rust
fn test_loss_landscape_stress_001() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }
```

### Group 5: 325 identical functions (e.g. `test_clipping_mod_stress_001` in `crates/brain-optim/src/clipping/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/clipping/mod.rs:98`):
```rust
fn test_clipping_mod_stress_001() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }
```

### Group 6: 322 identical functions (e.g. `test_schedulers_mod_stress_001` in `crates/brain-optim/src/schedulers/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/schedulers/mod.rs:123`):
```rust
fn test_schedulers_mod_stress_001() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }
```

### Group 7: 287 identical functions (e.g. `test_cosine_schedulers_stress_001` in `crates/brain-optim/src/schedulers/cosine.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/schedulers/cosine.rs:190`):
```rust
fn test_cosine_schedulers_stress_001() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }
```

### Group 8: 277 identical functions (e.g. `test_step_schedulers_stress_001` in `crates/brain-optim/src/schedulers/step.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/schedulers/step.rs:302`):
```rust
fn test_step_schedulers_stress_001() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }
```

### Group 9: 271 identical functions (e.g. `test_adaptive_clipping_stress_001` in `crates/brain-optim/src/clipping/adaptive.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/clipping/adaptive.rs:90`):
```rust
fn test_adaptive_clipping_stress_001() {
        let mut params = vec![Tensor::from_slice(&[1.0, 1.0], vec![2])];
        let mut grads = vec![Tensor::from_slice(&[1 as f64 * 10.0, (1 as f64) * 10.0], vec![2])];

        let agc = AGC::new(0.01, 1e-3);
        agc.clip(&mut params, &mut grads);

        let g0 = grads[0].data()[0];
        assert!(g0 < 100.0);
    }
```

### Group 10: 270 identical functions (e.g. `test_norm_clipping_stress_001` in `crates/brain-optim/src/clipping/norm.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/clipping/norm.rs:110`):
```rust
fn test_norm_clipping_stress_001() {
        let mut grads = vec![Tensor::from_slice(&[1 as f64, (1 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }
```

### Group 11: 265 identical functions (e.g. `test_nesterov_stress_001` in `crates/brain-optim/src/sgd/nesterov.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/sgd/nesterov.rs:167`):
```rust
fn test_nesterov_stress_001() {
        let group = ParamGroup::new(vec![0], 0.01);
        let mut opt = SgdNesterov::new(vec![group], 0.01, 0.9, 1e-4);

        let mut p = vec![Tensor::from_slice(&[1 as f64 * 0.2], vec![1])];
        let g = vec![Tensor::from_slice(&[0.5], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }
```

### Group 12: 263 identical functions (e.g. `test_adagrad_stress_001` in `crates/brain-optim/src/adagrad.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/adagrad.rs:185`):
```rust
fn test_adagrad_stress_001() {
        let group = ParamGroup::new(vec![0], 0.05);
        let mut opt = Adagrad::new(vec![group], AdagradConfig::default());

        let mut p = vec![Tensor::from_slice(&[1 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.2], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }
```

### Group 13: 263 identical functions (e.g. `test_lion_stress_001` in `crates/brain-optim/src/lion.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/lion.rs:186`):
```rust
fn test_lion_stress_001() {
        let group = ParamGroup::new(vec![0], 0.0001);
        let mut opt = Lion::new(vec![group], LionConfig::default());

        let mut p = vec![Tensor::from_slice(&[1 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.2], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }
```

### Group 14: 262 identical functions (e.g. `test_adadelta_stress_001` in `crates/brain-optim/src/adadelta.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/adadelta.rs:197`):
```rust
fn test_adadelta_stress_001() {
        let group = ParamGroup::new(vec![0], 1.0);
        let mut opt = Adadelta::new(vec![group], AdadeltaConfig::default());

        let mut p = vec![Tensor::from_slice(&[1 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.05], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }
```

### Group 15: 261 identical functions (e.g. `test_radam_stress_001` in `crates/brain-optim/src/radam.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/radam.rs:218`):
```rust
fn test_radam_stress_001() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = RAdam::new(vec![group], RAdamConfig::default());

        let mut p = vec![Tensor::from_slice(&[1 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.05], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }
```

### Group 16: 261 identical functions (e.g. `test_novograd_stress_001` in `crates/brain-optim/src/novograd.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/novograd.rs:211`):
```rust
fn test_novograd_stress_001() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = NovoGrad::new(vec![group], NovoGradConfig::default());

        let mut p = vec![Tensor::from_slice(&[1 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }
```

### Group 17: 260 identical functions (e.g. `test_lamb_stress_001` in `crates/brain-optim/src/lamb.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/lamb.rs:225`):
```rust
fn test_lamb_stress_001() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[1 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }
```

### Group 18: 259 identical functions (e.g. `test_rmsprop_stress_001` in `crates/brain-optim/src/rmsprop.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/rmsprop.rs:231`):
```rust
fn test_rmsprop_stress_001() {
        let group = ParamGroup::new(vec![0], 0.01);
        let mut opt = Rmsprop::new(vec![group], RmspropConfig::default());

        let mut p = vec![Tensor::from_slice(&[1 as f64 * 0.05], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }
```

### Group 19: 258 identical functions (e.g. `test_adam_stress_001` in `crates/brain-optim/src/adam/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/adam/mod.rs:248`):
```rust
fn test_adam_stress_001() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Adam::adamw(vec![group], 0.001, 1e-2);

        let mut p = vec![Tensor::from_slice(&[1 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }
```

### Group 20: 251 identical functions (e.g. `test_lookahead_stress_001` in `crates/brain-optim/src/lookahead.rs`)
- Files involved: 1
- Sample definition (`crates/brain-optim/src/lookahead.rs:83`):
```rust
fn test_lookahead_stress_001() {
        let mut lookahead = Lookahead::new(LookaheadConfig {
            k: 5,
            alpha: 0.5,
        });

        let mut p = vec![Tensor::from_slice(&[1 as f64], vec![1])];
        lookahead.init_slow_weights(&p);
        lookahead.step_lookahead(&mut p);
        assert_eq!(lookahead.step_count, 1);
    }
```
