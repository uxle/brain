# Test Duplication & Inflation Audit Report

- **Target Directory:** `crates/brain-regularization/src`
- **Total Test Functions Scanned:** 7976
- **Duplicate / Template Groups:** 27
- **Total Padded / Duplicate Test Functions:** 7976 (100.0% of total tests)
- **Redundant Functions Removable:** 7949

## Summary by File

| File | Total Tests | Duplicated Tests | Redundancy Ratio |
|---|---|---|---|
| `crates/brain-regularization/src/augment.rs` | 228 | 228 | 100.0% |
| `crates/brain-regularization/src/config.rs` | 231 | 231 | 100.0% |
| `crates/brain-regularization/src/consistency.rs` | 409 | 409 | 100.0% |
| `crates/brain-regularization/src/core.rs` | 292 | 292 | 100.0% |
| `crates/brain-regularization/src/curriculum.rs` | 252 | 252 | 100.0% |
| `crates/brain-regularization/src/decay.rs` | 409 | 409 | 100.0% |
| `crates/brain-regularization/src/dropout/adaptive.rs` | 358 | 358 | 100.0% |
| `crates/brain-regularization/src/dropout/alpha.rs` | 403 | 403 | 100.0% |
| `crates/brain-regularization/src/dropout/mod.rs` | 264 | 264 | 100.0% |
| `crates/brain-regularization/src/dropout_uncertainty.rs` | 269 | 269 | 100.0% |
| `crates/brain-regularization/src/earlystop.rs` | 201 | 201 | 100.0% |
| `crates/brain-regularization/src/impl.rs` | 329 | 329 | 100.0% |
| `crates/brain-regularization/src/label_smooth.rs` | 250 | 250 | 100.0% |
| `crates/brain-regularization/src/lib.rs` | 464 | 464 | 100.0% |
| `crates/brain-regularization/src/normalization/batch.rs` | 166 | 166 | 100.0% |
| `crates/brain-regularization/src/normalization/group.rs` | 224 | 224 | 100.0% |
| `crates/brain-regularization/src/normalization/layer.rs` | 185 | 185 | 100.0% |
| `crates/brain-regularization/src/normalization/mod.rs` | 413 | 413 | 100.0% |
| `crates/brain-regularization/src/normalization/weight.rs` | 243 | 243 | 100.0% |
| `crates/brain-regularization/src/ops.rs` | 299 | 299 | 100.0% |
| `crates/brain-regularization/src/perturb.rs` | 270 | 270 | 100.0% |
| `crates/brain-regularization/src/registry.rs` | 411 | 411 | 100.0% |
| `crates/brain-regularization/src/regularizers.rs` | 199 | 199 | 100.0% |
| `crates/brain-regularization/src/rules.rs` | 298 | 298 | 100.0% |
| `crates/brain-regularization/src/stopping.rs` | 270 | 270 | 100.0% |
| `crates/brain-regularization/src/train_hooks.rs` | 409 | 409 | 100.0% |
| `crates/brain-regularization/src/utils.rs` | 230 | 230 | 100.0% |

## Top Duplicate Groups

### Group 1: 464 identical functions (e.g. `test_lib_root_stress_001` in `crates/brain-regularization/src/lib.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/lib.rs:98`):
```rust
fn test_lib_root_stress_001() {
        assert_eq!(VERSION, "0.2.0");
        let reg = RegRegistry::parse_kind("dropout").unwrap();
        assert_eq!(reg, RegKind::Dropout);
    }
```

### Group 2: 413 identical functions (e.g. `test_normalization_mod_stress_001` in `crates/brain-regularization/src/normalization/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/normalization/mod.rs:44`):
```rust
fn test_normalization_mod_stress_001() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }
```

### Group 3: 411 identical functions (e.g. `test_registry_stress_001` in `crates/brain-regularization/src/registry.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/registry.rs:60`):
```rust
fn test_registry_stress_001() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_1").is_err());
    }
```

### Group 4: 409 identical functions (e.g. `test_decay_stress_001` in `crates/brain-regularization/src/decay.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/decay.rs:72`):
```rust
fn test_decay_stress_001() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 1 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }
```

### Group 5: 409 identical functions (e.g. `test_consistency_stress_001` in `crates/brain-regularization/src/consistency.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/consistency.rs:71`):
```rust
fn test_consistency_stress_001() {
        let p1 = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let p2 = Tensor::from_slice(&[1.1, 1.9], vec![2]);
        let loss = compute_consistency_loss(&p1, &p2, 1.0).unwrap();
        assert!(loss > 0.0 && loss < 0.1);
    }
```

### Group 6: 409 identical functions (e.g. `test_train_hooks_stress_001` in `crates/brain-regularization/src/train_hooks.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/train_hooks.rs:77`):
```rust
fn test_train_hooks_stress_001() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 1 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }
```

### Group 7: 403 identical functions (e.g. `test_alpha_dropout_stress_001` in `crates/brain-regularization/src/dropout/alpha.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/dropout/alpha.rs:124`):
```rust
fn test_alpha_dropout_stress_001() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 1 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }
```

### Group 8: 358 identical functions (e.g. `test_adaptive_dropout_stress_001` in `crates/brain-regularization/src/dropout/adaptive.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/dropout/adaptive.rs:120`):
```rust
fn test_adaptive_dropout_stress_001() {
        let mut cd = ConcreteDropout::new(0.3, 0.1);
        let t = Tensor::from_slice(&[1.0, 2.0, 1 as f64 * 0.1, 4.0], vec![4]);
        let out = cd.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
        assert!(cd.current_p() > 0.0 && cd.current_p() < 1.0);
    }
```

### Group 9: 329 identical functions (e.g. `test_impl_stress_001` in `crates/brain-regularization/src/impl.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/impl.rs:58`):
```rust
fn test_impl_stress_001() {
        let t = Tensor::from_slice(&[1.0, 2.0, 1 as f64 * 0.1, 4.0], vec![4]);
        let d_out = apply_dropout(&t, 0.5, false).unwrap();
        assert_eq!(d_out.data(), t.data());

        let ln_out = apply_layernorm(&t, vec![4], 1e-5).unwrap();
        assert_eq!(ln_out.shape(), &[4]);
    }
```

### Group 10: 299 identical functions (e.g. `test_ops_stress_001` in `crates/brain-regularization/src/ops.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/ops.rs:54`):
```rust
fn test_ops_stress_001() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }
```

### Group 11: 298 identical functions (e.g. `test_rules_stress_001` in `crates/brain-regularization/src/rules.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/rules.rs:72`):
```rust
fn test_rules_stress_001() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 1 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }
```

### Group 12: 292 identical functions (e.g. `test_core_stress_001` in `crates/brain-regularization/src/core.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/core.rs:135`):
```rust
fn test_core_stress_001() {
        let mut state = RegState::default();
        state.step_count = 1;
        assert!(state.is_training);
        assert_eq!(state.step_count, 1);

        let err = RegError::InvalidProbability(1.5);
        assert!(err.to_string().contains("1.5"));
    }
```

### Group 13: 270 identical functions (e.g. `test_stopping_stress_001` in `crates/brain-regularization/src/stopping.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/stopping.rs:100`):
```rust
fn test_stopping_stress_001() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 1, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }
```

### Group 14: 270 identical functions (e.g. `test_perturb_stress_001` in `crates/brain-regularization/src/perturb.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/perturb.rs:107`):
```rust
fn test_perturb_stress_001() {
        let mut gn = GaussianNoise::new(0.05);
        let t = Tensor::from_slice(&[1.0, 2.0, 1 as f64 * 0.1], vec![3]);
        let noisy = gn.inject(&t);
        assert_eq!(noisy.shape(), &[3]);

        let grad = Tensor::from_slice(&[0.1, -0.2, 0.0], vec![3]);
        let fgsm = apply_fgsm_perturbation(&t, &grad, 0.1).unwrap();
        assert_eq!(fgsm.shape(), &[3]);
    }
```

### Group 15: 269 identical functions (e.g. `test_dropout_uncertainty_stress_001` in `crates/brain-regularization/src/dropout_uncertainty.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/dropout_uncertainty.rs:112`):
```rust
fn test_dropout_uncertainty_stress_001() {
        let s1 = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let s2 = Tensor::from_slice(&[1.2, 1.8], vec![2]);
        let s3 = Tensor::from_slice(&[0.8, 2.2 + (1 as f64 * 0.001)], vec![2]);

        let res = compute_mc_dropout_statistics(&[s1, s2, s3]).unwrap();
        assert_eq!(res.mean.shape(), &[2]);
        assert_eq!(res.variance.shape(), &[2]);
        assert_eq!(res.std_dev.shape(), &[2]);
    }
```

### Group 16: 264 identical functions (e.g. `test_dropout_stress_001` in `crates/brain-regularization/src/dropout/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/dropout/mod.rs:182`):
```rust
fn test_dropout_stress_001() {
        let mut drop = Dropout::with_seed(0.5, 1 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }
```

### Group 17: 252 identical functions (e.g. `test_curriculum_stress_001` in `crates/brain-regularization/src/curriculum.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/curriculum.rs:73`):
```rust
fn test_curriculum_stress_001() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 1), 1.0);
    }
```

### Group 18: 250 identical functions (e.g. `test_label_smooth_stress_001` in `crates/brain-regularization/src/label_smooth.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/label_smooth.rs:94`):
```rust
fn test_label_smooth_stress_001() {
        let ls = LabelSmoothing::new(LabelSmoothConfig {
            smoothing: 0.1,
            num_classes: 2,
        });
        let targets = vec![0, 1];
        let smoothed = ls.smooth_targets(&targets).unwrap();
        assert_eq!(smoothed.shape(), &[2, 2]);
        assert!((smoothed.data()[0] - 0.95).abs() < 1e-6);
        assert!((smoothed.data()[1] - 0.05).abs() < 1e-6);
    }
```

### Group 19: 243 identical functions (e.g. `test_weight_norm_stress_001` in `crates/brain-regularization/src/normalization/weight.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/normalization/weight.rs:180`):
```rust
fn test_weight_norm_stress_001() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (1 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }
```

### Group 20: 231 identical functions (e.g. `test_config_stress_001` in `crates/brain-regularization/src/config.rs`)
- Files involved: 1
- Sample definition (`crates/brain-regularization/src/config.rs:116`):
```rust
fn test_config_stress_001() {
        let cfg = RegConfig::default();
        cfg.validate().unwrap();
        assert_eq!(cfg.dropout_p, 0.5);

        let d_cfg = DropoutConfig {
            p: (1 as f64 * 0.001).min(0.99),
            in_place: false,
            seed: Some(1 as u64),
        };
        assert!(d_cfg.p >= 0.0 && d_cfg.p < 1.0);
    }
```
