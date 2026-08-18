//! # Closed-Form Optimizer Step Verification Tests

use brain_core::Tensor;
use brain_optim::prelude::*;
use brain_optim::schedulers::{LrScheduler, StepLR, CosineAnnealingLR};
use brain_optim::state::StateDict;
use brain_optim::ParamGroup;

#[test]
fn test_sgd_closed_form_step() {
    // Minimize f(x) = x^2 starting from x = 1.0
    // grad = 2x = 2.0. With lr = 0.1, x_new = 1.0 - 0.1 * 2.0 = 0.8 exactly.
    let mut params = vec![Tensor::from_vec(vec![1.0], vec![1])];
    let grads = vec![Tensor::from_vec(vec![2.0], vec![1])];

    let group = ParamGroup::new(vec![0], 0.1);
    let mut opt = Sgd::new(vec![group], SgdConfig {
        lr: 0.1,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        decoupled_weight_decay: false,
    });

    opt.step(&mut params, &grads).unwrap();
    let x_val = params[0].get(0);
    assert!((x_val - 0.8).abs() < 1e-9, "SGD closed form expected 0.8, got {}", x_val);
}

#[test]
fn test_adam_closed_form_step() {
    // Step 1 on g = 2.0, lr = 0.1, beta1 = 0.9, beta2 = 0.999
    // m1 = 0.2, v1 = 0.004 -> m_hat = 2.0, v_hat = 4.0 -> delta = 0.1 -> x_new = 0.9
    let mut params = vec![Tensor::from_vec(vec![1.0], vec![1])];
    let grads = vec![Tensor::from_vec(vec![2.0], vec![1])];

    let group = ParamGroup::new(vec![0], 0.1);
    let mut opt = Adam::new(vec![group], AdamConfig {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        amsgrad: false,
        decoupled_weight_decay: false,
    });

    opt.step(&mut params, &grads).unwrap();
    let x_val = params[0].get(0);
    assert!((x_val - 0.9).abs() < 1e-6, "Adam closed form expected 0.9, got {}", x_val);
}

#[test]
fn test_adamw_closed_form_step() {
    // Step 1 on g = 2.0, lr = 0.1, weight_decay = 0.01
    // Adam update = -0.1, decoupled weight decay = -lr * wd * x = -0.1 * 0.01 * 1.0 = -0.001
    // x_new = 1.0 - 0.001 - 0.1 = 0.899
    let mut params = vec![Tensor::from_vec(vec![1.0], vec![1])];
    let grads = vec![Tensor::from_vec(vec![2.0], vec![1])];

    let group = ParamGroup::new(vec![0], 0.1);
    let mut opt = Adam::adamw(vec![group], 0.1, 0.01);

    opt.step(&mut params, &grads).unwrap();
    let x_val = params[0].get(0);
    assert!((x_val - 0.899).abs() < 1e-6, "AdamW closed form expected 0.899, got {}", x_val);
}

#[test]
fn test_state_dict_round_trip() {
    let mut sd = StateDict::new("Adam", 42);
    sd.insert_scalar("lr", 0.05);
    sd.insert_tensor("exp_avg.0", Tensor::from_vec(vec![0.1, 0.2, 0.3], vec![3]));

    let bytes = sd.save_bytes();
    let loaded = StateDict::from_bytes(&bytes).unwrap();

    assert_eq!(loaded.metadata.optimizer_type, "Adam");
    assert_eq!(loaded.metadata.step, 42);
    assert_eq!(loaded.get_scalar("lr"), Some(0.05));
    let t = loaded.get_tensor("exp_avg.0").unwrap();
    assert_eq!(t.shape(), &[3]);
    assert_eq!(t.to_vec(), vec![0.1, 0.2, 0.3]);
}

#[test]
fn test_lr_scheduler_step() {
    let group = ParamGroup::new(vec![0], 0.1);
    let mut opt = Sgd::with_lr(vec![group], 0.1);

    let mut scheduler = StepLR::new(vec![0.1], 10, 0.5);
    for _ in 0..10 {
        let _ = scheduler.step(&mut opt);
    }
    assert!((opt.get_lr() - 0.05).abs() < 1e-9, "Expected lr=0.05 after 10 steps, got {}", opt.get_lr());

    let mut cosine = CosineAnnealingLR::new(vec![0.1], 20, 0.0);
    opt.set_lr(0.1);
    for _ in 0..20 {
        let _ = cosine.step(&mut opt);
    }
    assert!(opt.get_lr() < 1e-5, "Expected lr near 0.0 after full cosine period, got {}", opt.get_lr());
}
