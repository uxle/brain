# Test Duplication & Inflation Audit Report

- **Target Directory:** `crates/brain-loss/src`
- **Total Test Functions Scanned:** 8366
- **Duplicate / Template Groups:** 26
- **Total Padded / Duplicate Test Functions:** 8366 (100.0% of total tests)
- **Redundant Functions Removable:** 8340

## Summary by File

| File | Total Tests | Duplicated Tests | Redundancy Ratio |
|---|---|---|---|
| `crates/brain-loss/src/adversarial/mod.rs` | 301 | 301 | 100.0% |
| `crates/brain-loss/src/adversarial/other.rs` | 219 | 219 | 100.0% |
| `crates/brain-loss/src/adversarial/wasserstein.rs` | 367 | 367 | 100.0% |
| `crates/brain-loss/src/classification/ce.rs` | 327 | 327 | 100.0% |
| `crates/brain-loss/src/classification/focal.rs` | 408 | 408 | 100.0% |
| `crates/brain-loss/src/classification/mod.rs` | 330 | 330 | 100.0% |
| `crates/brain-loss/src/classification/other.rs` | 232 | 232 | 100.0% |
| `crates/brain-loss/src/combine.rs` | 364 | 364 | 100.0% |
| `crates/brain-loss/src/config.rs` | 298 | 298 | 100.0% |
| `crates/brain-loss/src/contrastive/infonce.rs` | 328 | 328 | 100.0% |
| `crates/brain-loss/src/contrastive/mod.rs` | 275 | 275 | 100.0% |
| `crates/brain-loss/src/contrastive/simclr.rs` | 364 | 364 | 100.0% |
| `crates/brain-loss/src/contrastive/triplet.rs` | 297 | 297 | 100.0% |
| `crates/brain-loss/src/core.rs` | 406 | 406 | 100.0% |
| `crates/brain-loss/src/distillation.rs` | 363 | 363 | 100.0% |
| `crates/brain-loss/src/impl_.rs` | 326 | 326 | 100.0% |
| `crates/brain-loss/src/masked.rs` | 414 | 414 | 100.0% |
| `crates/brain-loss/src/metric_loss/mod.rs` | 407 | 407 | 100.0% |
| `crates/brain-loss/src/ops.rs` | 203 | 203 | 100.0% |
| `crates/brain-loss/src/regression/dirichlet.rs` | 363 | 363 | 100.0% |
| `crates/brain-loss/src/regression/mod.rs` | 367 | 367 | 100.0% |
| `crates/brain-loss/src/regression/mse.rs` | 204 | 204 | 100.0% |
| `crates/brain-loss/src/regression/robust.rs` | 251 | 251 | 100.0% |
| `crates/brain-loss/src/segmentation/ce_dice.rs` | 363 | 363 | 100.0% |
| `crates/brain-loss/src/segmentation/mod.rs` | 370 | 370 | 100.0% |
| `crates/brain-loss/src/utils.rs` | 219 | 219 | 100.0% |

## Top Duplicate Groups

### Group 1: 414 identical functions (e.g. `test_masked_stress_001` in `crates/brain-loss/src/masked.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/masked.rs:34`):
```rust
fn test_masked_stress_001() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }
```

### Group 2: 408 identical functions (e.g. `test_focal_stress_001` in `crates/brain-loss/src/classification/focal.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/classification/focal.rs:82`):
```rust
fn test_focal_stress_001() {
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let fl = FocalLoss::new(FocalConfig::default());
        let loss = fl.compute(&logits, &[2]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);
    }
```

### Group 3: 407 identical functions (e.g. `test_metric_loss_stress_001` in `crates/brain-loss/src/metric_loss/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/metric_loss/mod.rs:83`):
```rust
fn test_metric_loss_stress_001() {
        let cos_t = Tensor::from_vec(vec![0.9, 0.1, 0.1], vec![1, 3]);
        let arc = ArcFaceLoss::default();
        let l = arc.compute(&cos_t, &[0]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }
```

### Group 4: 406 identical functions (e.g. `test_core_stress_001` in `crates/brain-loss/src/core.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/core.rs:102`):
```rust
fn test_core_stress_001() {
        let lv = LossValue::new(1 as f64 * 0.1);
        assert!((lv.scalar - 1 as f64 * 0.1).abs() < 1e-9);
        assert_eq!(Reduction::default(), Reduction::Mean);
        assert_eq!(LossKind::default(), LossKind::CrossEntropy);
    }
```

### Group 5: 370 identical functions (e.g. `test_seg_mod_stress_001` in `crates/brain-loss/src/segmentation/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/segmentation/mod.rs:16`):
```rust
fn test_seg_mod_stress_001() {
        let loss = CEDiceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let l = loss.compute(&p, &t).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }
```

### Group 6: 367 identical functions (e.g. `test_reg_mod_stress_001` in `crates/brain-loss/src/regression/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/regression/mod.rs:43`):
```rust
fn test_reg_mod_stress_001() {
        let mse = MSELoss::default();
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let l = mse.compute(&p, &t).unwrap();
        assert!((l.to_vec()[0] - 0.5).abs() < 1e-9);
    }
```

### Group 7: 367 identical functions (e.g. `test_wasserstein_stress_001` in `crates/brain-loss/src/adversarial/wasserstein.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/adversarial/wasserstein.rs:42`):
```rust
fn test_wasserstein_stress_001() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let f = Tensor::from_vec(vec![0.0, 1.0], vec![2]);
        let d_l = wgan.discriminator_loss(&r, &f).unwrap();
        assert!((d_l.to_vec()[0] - -2.0).abs() < 1e-9);
    }
```

### Group 8: 364 identical functions (e.g. `test_combine_stress_001` in `crates/brain-loss/src/combine.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/combine.rs:70`):
```rust
fn test_combine_stress_001() {
        let l1 = Tensor::from_vec(vec![2.0], vec![1]);
        let l2 = Tensor::from_vec(vec![3.0], vec![1]);
        let comp = CompositeLoss::new(vec![0.5, 0.5]);
        let res = comp.combine(&[l1, l2]).unwrap();
        assert!((res.to_vec()[0] - 2.5).abs() < 1e-9);
    }
```

### Group 9: 364 identical functions (e.g. `test_simclr_stress_001` in `crates/brain-loss/src/contrastive/simclr.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/contrastive/simclr.rs:73`):
```rust
fn test_simclr_stress_001() {
        let z1 = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let z2 = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let scl = SimCLRLoss::default();
        let loss = scl.compute(&z1, &z2).unwrap();
        assert!(loss.to_vec()[0] > 0.0);
    }
```

### Group 10: 363 identical functions (e.g. `test_distill_stress_001` in `crates/brain-loss/src/distillation.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/distillation.rs:75`):
```rust
fn test_distill_stress_001() {
        let s = Tensor::from_vec(vec![2.0, 1.0], vec![1, 2]);
        let t = Tensor::from_vec(vec![2.1, 0.9], vec![1, 2]);
        let kd = KnowledgeDistillationLoss::default();
        let l = kd.compute(&s, &t).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }
```

### Group 11: 363 identical functions (e.g. `test_dirichlet_stress_001` in `crates/brain-loss/src/regression/dirichlet.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/regression/dirichlet.rs:79`):
```rust
fn test_dirichlet_stress_001() {
        let x1 = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let x2 = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let cos_loss = CosineEmbeddingLoss::default();
        let l = cos_loss.compute(&x1, &x2, &[1.0]).unwrap();
        assert!(l.to_vec()[0].abs() < 1e-9);
    }
```

### Group 12: 363 identical functions (e.g. `test_ce_dice_stress_001` in `crates/brain-loss/src/segmentation/ce_dice.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/segmentation/ce_dice.rs:75`):
```rust
fn test_ce_dice_stress_001() {
        let p = Tensor::from_vec(vec![0.8, 0.9, 0.1, 0.05], vec![4]);
        let t = Tensor::from_vec(vec![1.0, 1.0, 0.0, 0.0], vec![4]);
        let loss_fn = CEDiceLoss::default();
        let l = loss_fn.compute(&p, &t).unwrap();
        assert!(l.to_vec()[0] >= 0.0);
    }
```

### Group 13: 330 identical functions (e.g. `test_class_mod_stress_001` in `crates/brain-loss/src/classification/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/classification/mod.rs:56`):
```rust
fn test_class_mod_stress_001() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }
```

### Group 14: 328 identical functions (e.g. `test_infonce_stress_001` in `crates/brain-loss/src/contrastive/infonce.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/contrastive/infonce.rs:70`):
```rust
fn test_infonce_stress_001() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }
```

### Group 15: 327 identical functions (e.g. `test_ce_stress_001` in `crates/brain-loss/src/classification/ce.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/classification/ce.rs:117`):
```rust
fn test_ce_stress_001() {
        let logits = Tensor::from_vec(vec![0.5, 1.5, 2.5, 3.0, 1.0, 0.1], vec![2, 3]);
        let mut cfg = CrossEntropyConfig::default();
        cfg.label_smoothing = 0.1;
        let ce = CrossEntropyLoss::new(cfg);
        let loss = ce.compute(&logits, &[2, 0]).unwrap();
        assert!(loss.to_vec()[0] > 0.0);
    }
```

### Group 16: 326 identical functions (e.g. `test_impl_stress_001` in `crates/brain-loss/src/impl_.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/impl_.rs:82`):
```rust
fn test_impl_stress_001() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }
```

### Group 17: 301 identical functions (e.g. `test_adv_mod_stress_001` in `crates/brain-loss/src/adversarial/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/adversarial/mod.rs:37`):
```rust
fn test_adv_mod_stress_001() {
        let wgan = WassersteinLoss::default();
        let r = Tensor::from_vec(vec![1.0], vec![1]);
        let f = Tensor::from_vec(vec![-1.0], vec![1]);
        let dl = wgan.discriminator_loss(&r, &f).unwrap();
        let gl = wgan.generator_loss(&f).unwrap();
        assert!((dl.to_vec()[0] - -2.0).abs() < 1e-9);
        assert!((gl.to_vec()[0] - 1.0).abs() < 1e-9);
    }
```

### Group 18: 298 identical functions (e.g. `test_config_stress_001` in `crates/brain-loss/src/config.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/config.rs:68`):
```rust
fn test_config_stress_001() {
        let mut cfg = LossConfig::default();
        cfg.delta = 1 as f64 * 0.1 + 0.1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.delta = -1.0;
        assert!(cfg.validate().is_err());
    }
```

### Group 19: 297 identical functions (e.g. `test_triplet_stress_001` in `crates/brain-loss/src/contrastive/triplet.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/contrastive/triplet.rs:73`):
```rust
fn test_triplet_stress_001() {
        let a = Tensor::from_vec(vec![0.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![0.1, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![2.0, 0.0], vec![1, 2]);

        let tl = TripletMarginLoss::default();
        let l = tl.compute(&a, &p, &n).unwrap();
        assert_eq!(l.to_vec()[0], 0.0); // 0.1 - 2.0 + 1.0 = -0.9 -> 0.0
    }
```

### Group 20: 275 identical functions (e.g. `test_contrastive_mod_stress_001` in `crates/brain-loss/src/contrastive/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-loss/src/contrastive/mod.rs:43`):
```rust
fn test_contrastive_mod_stress_001() {
        let cfg = ContrastiveConfig::default();
        assert_eq!(cfg.temperature, 0.07);
        let trip = TripletMarginLoss::default();
        let a = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![0.9, 0.1], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let l = trip.compute(&a, &p, &n).unwrap();
        assert!(l.to_vec()[0] >= 0.0);
    }
```
