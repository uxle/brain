//! # Closed-Form Optimizer Multi-Step Reference Trajectories & Invariants

use brain_core::Tensor;
use brain_optim::clipping::{clip_grad_norm_, NormType};
use brain_optim::schedulers::{CosineAnnealingLR, LinearWarmup, LrScheduler, StepLR};
use brain_optim::state::StateDict;
use brain_optim::ParamGroup;
use brain_optim::*;

// =============================================================================
// 1. Exact 5-Step Trajectories: SGD, Momentum, Nesterov
// =============================================================================

#[test]
fn test_sgd_standard_momentum_5_step_trajectory() {
    // Loss L(theta) = theta^2, grad(theta) = 2*theta
    // Parameters: theta_0 = 1.0, lr = 0.1, momentum = 0.9, wd = 0.0
    let mut params = vec![Tensor::from_slice(&[1.0], vec![1])];
    let group = ParamGroup::new(vec![0], 0.1);
    let mut opt = Sgd::new(
        vec![group],
        SgdConfig {
            lr: 0.1,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            decoupled_weight_decay: false,
        },
    );

    // Hand-calculated exact 5-step trajectory:
    let expected_thetas = [
        0.8,      // Step 1: g=2.0, v=2.0, theta = 1.0 - 0.2 = 0.8
        0.46,     // Step 2: g=1.6, v=0.9(2.0)+1.6=3.4, theta = 0.8 - 0.34 = 0.46
        0.062,    // Step 3: g=0.92, v=0.9(3.4)+0.92=3.98, theta = 0.46 - 0.398 = 0.062
        -0.3086,  // Step 4: g=0.124, v=0.9(3.98)+0.124=3.706, theta = 0.062 - 0.3706 = -0.3086
        -0.58042, // Step 5: g=-0.6172, v=0.9(3.706)-0.6172=2.7182, theta = -0.3086 - 0.27182 = -0.58042
    ];

    for (step, &expected) in expected_thetas.iter().enumerate() {
        let current_theta = params[0].get(0);
        let grad = Tensor::from_slice(&[2.0 * current_theta], vec![1]);
        opt.step(&mut params, &[grad]).unwrap();
        let updated_theta = params[0].get(0);
        assert!(
            (updated_theta - expected).abs() < 1e-10,
            "Step {} SGD momentum mismatch: expected={:.8}, got={:.8}",
            step + 1,
            expected,
            updated_theta
        );
    }
}

#[test]
fn test_sgd_nesterov_momentum_5_step_trajectory() {
    let mut params = vec![Tensor::from_slice(&[1.0], vec![1])];
    let group = ParamGroup::new(vec![0], 0.1);
    let mut opt = Sgd::new(
        vec![group],
        SgdConfig {
            lr: 0.1,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: true,
            decoupled_weight_decay: false,
        },
    );

    let expected_thetas = [
        0.62,          // Step 1
        0.2224,        // Step 2
        -0.108352,     // Step 3
        -0.32482304,   // Step 4
        -0.4157175808, // Step 5
    ];

    for (step, &expected) in expected_thetas.iter().enumerate() {
        let current_theta = params[0].get(0);
        let grad = Tensor::from_slice(&[2.0 * current_theta], vec![1]);
        opt.step(&mut params, &[grad]).unwrap();
        let updated_theta = params[0].get(0);
        assert!(
            (updated_theta - expected).abs() < 1e-10,
            "Step {} SGD Nesterov mismatch: expected={:.8}, got={:.8}",
            step + 1,
            expected,
            updated_theta
        );
    }
}

// =============================================================================
// 2. Exact Trajectories: Adam, AdamW, RMSProp, Adagrad
// =============================================================================

#[test]
fn test_adam_multi_step_trajectory() {
    let mut params = vec![Tensor::from_slice(&[1.0], vec![1])];
    let group = ParamGroup::new(vec![0], 0.1);
    let mut opt = Adam::new(
        vec![group],
        AdamConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            decoupled_weight_decay: false,
        },
    );

    // Step 1: theta_0 = 1.0, g_1 = 2.0 -> theta_1 = 0.9 exactly
    let g1 = Tensor::from_slice(&[2.0 * params[0].get(0)], vec![1]);
    opt.step(&mut params, &[g1]).unwrap();
    assert!((params[0].get(0) - 0.9).abs() < 1e-6);

    // Step 2: g_2 = 1.8
    // m_2 = 0.9(0.2) + 0.1(1.8) = 0.18 + 0.18 = 0.36 -> m_hat = 0.36 / (1 - 0.81) = 1.8947368
    // v_2 = 0.999(0.004) + 0.001(3.24) = 0.003996 + 0.00324 = 0.007236 -> v_hat = 0.007236 / (1 - 0.999^2) = 3.6216216
    // delta = 0.1 * 1.8947368 / sqrt(3.6216216) = 0.099562
    // theta_2 = 0.9 - 0.099562 = 0.800438
    let g2 = Tensor::from_slice(&[2.0 * params[0].get(0)], vec![1]);
    opt.step(&mut params, &[g2]).unwrap();
    assert!((params[0].get(0) - 0.800438).abs() < 1e-4);
}

#[test]
fn test_adamw_decoupled_weight_decay_exact() {
    let mut params = vec![Tensor::from_slice(&[1.0], vec![1])];
    let group = ParamGroup::new(vec![0], 0.1);
    let mut opt = Adam::adamw(vec![group], 0.1, 0.05);

    // Step 1: g_1 = 2.0
    // Decoupled weight decay: theta_0 * (1 - lr * wd) = 1.0 * (1 - 0.1 * 0.05) = 0.995
    // Adam update: -0.1
    // theta_1 = 0.995 - 0.1 = 0.895
    let g1 = Tensor::from_slice(&[2.0], vec![1]);
    opt.step(&mut params, &[g1]).unwrap();
    assert!((params[0].get(0) - 0.895).abs() < 1e-6);
}

#[test]
fn test_rmsprop_step_trajectory() {
    let mut params = vec![Tensor::from_slice(&[1.0], vec![1])];
    let group = ParamGroup::new(vec![0], 0.1);
    let mut opt = Rmsprop::new(
        vec![group],
        RmspropConfig {
            lr: 0.1,
            alpha: 0.9,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
        },
    );

    // Step 1: theta_0 = 1.0, g_1 = 2.0
    // v_1 = (1 - alpha) * g_1^2 = 0.1 * 4.0 = 0.4
    // theta_1 = 1.0 - 0.1 * 2.0 / sqrt(0.4) = 1.0 - 0.2 / 0.63245553 = 1.0 - 0.31622776 = 0.6837722
    let g1 = Tensor::from_slice(&[2.0], vec![1]);
    opt.step(&mut params, &[g1]).unwrap();
    assert!((params[0].get(0) - 0.6837722).abs() < 1e-6);
}

#[test]
fn test_adagrad_step_trajectory() {
    let mut params = vec![Tensor::from_slice(&[1.0], vec![1])];
    let group = ParamGroup::new(vec![0], 0.1);
    let mut opt = Adagrad::new(
        vec![group],
        AdagradConfig {
            lr: 0.1,
            lr_decay: 0.0,
            weight_decay: 0.0,
            initial_accumulator_value: 0.0,
            eps: 1e-10,
        },
    );

    // Step 1: theta_0 = 1.0, g_1 = 2.0
    // v_1 = 4.0
    // theta_1 = 1.0 - 0.1 * 2.0 / sqrt(4.0) = 1.0 - 0.1 = 0.9
    let g1 = Tensor::from_slice(&[2.0], vec![1]);
    opt.step(&mut params, &[g1]).unwrap();
    assert!((params[0].get(0) - 0.9).abs() < 1e-6);

    // Step 2: g_2 = 1.8
    // v_2 = 4.0 + 3.24 = 7.24
    // theta_2 = 0.9 - 0.1 * 1.8 / sqrt(7.24) = 0.9 - 0.18 / 2.6907248 = 0.9 - 0.066896 = 0.8331035
    let g2 = Tensor::from_slice(&[1.8], vec![1]);
    opt.step(&mut params, &[g2]).unwrap();
    assert!((params[0].get(0) - 0.8331035).abs() < 1e-6);
}

// =============================================================================
// 3. State Management & Edge Cases
// =============================================================================

#[test]
fn test_multi_param_group_independent_stepping() {
    let mut params = vec![
        Tensor::from_slice(&[1.0], vec![1]),
        Tensor::from_slice(&[1.0], vec![1]),
    ];
    let group1 = ParamGroup::new(vec![0], 0.1);
    let group2 = ParamGroup::new(vec![1], 0.01);

    let mut opt = Sgd::new(
        vec![group1, group2],
        SgdConfig {
            lr: 0.1,
            ..Default::default()
        },
    );

    let grads = vec![
        Tensor::from_slice(&[2.0], vec![1]),
        Tensor::from_slice(&[2.0], vec![1]),
    ];

    opt.step(&mut params, &grads).unwrap();

    // Param 0 updated with lr=0.1 -> 1.0 - 0.2 = 0.8
    assert!((params[0].get(0) - 0.8).abs() < 1e-9);
    // Param 1 updated with lr=0.01 -> 1.0 - 0.02 = 0.98
    assert!((params[1].get(0) - 0.98).abs() < 1e-9);

    // Step count increments once per overall step call
    assert_eq!(opt.step_count, 1);
}

#[test]
fn test_zero_gradient_numerical_stability() {
    let mut params = vec![Tensor::from_slice(&[5.0], vec![1])];
    let grads = vec![Tensor::from_slice(&[0.0], vec![1])];

    let group = ParamGroup::new(vec![0], 0.1);
    let mut opt = Adam::new(vec![group], AdamConfig::default());

    opt.step(&mut params, &grads).unwrap();
    assert_eq!(
        params[0].get(0),
        5.0,
        "Zero gradient should leave parameter unchanged without NaN"
    );
    assert!(!params[0].get(0).is_nan());
}

// =============================================================================
// 4. Scheduler Boundary Conditions
// =============================================================================

#[test]
fn test_scheduler_exact_boundary_conditions() {
    let group = ParamGroup::new(vec![0], 0.1);
    let mut opt = Sgd::with_lr(vec![group], 0.1);

    // StepLR at step_size = 5
    let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
    for s in 1..=5 {
        step_lr.step(&mut opt).unwrap();
        if s < 5 {
            assert_eq!(opt.get_lr(), 0.1, "LR must remain 0.1 before step 5");
        } else {
            assert_eq!(opt.get_lr(), 0.05, "LR must step to 0.05 exactly at step 5");
        }
    }

    // CosineAnnealingLR endpoints
    let mut cosine = CosineAnnealingLR::new(vec![0.1], 10, 0.001);
    opt.set_lr(0.1);
    for _ in 0..10 {
        cosine.step(&mut opt).unwrap();
    }
    assert!(
        (opt.get_lr() - 0.001).abs() < 1e-6,
        "Cosine must reach eta_min at step T_max"
    );

    // LinearWarmup
    let mut warmup = LinearWarmup::new(vec![0.1], vec![0.0], 5);
    opt.set_lr(0.0);
    for step in 1..=5 {
        warmup.step(&mut opt).unwrap();
        let expected = 0.1 * (step as f64 / 5.0);
        assert!((opt.get_lr() - expected).abs() < 1e-9);
    }
}

// =============================================================================
// 5. Global Norm Gradient Clipping Across Multi-Shape Tensors
// =============================================================================

#[test]
fn test_clip_grad_norm_multi_tensor_global() {
    let g1 = Tensor::from_slice(&[3.0, 4.0], vec![2]); // norm = 5.0
    let g2 = Tensor::from_slice(&[0.0, 12.0], vec![1, 2]); // norm = 12.0
                                                           // Total global Euclidean norm = sqrt(5^2 + 12^2) = sqrt(25 + 144) = sqrt(169) = 13.0

    let mut grads = vec![g1, g2];
    let total_norm = clip_grad_norm_(&mut grads, 6.5, NormType::L2);
    assert!((total_norm - 13.0).abs() < 1e-9);

    // Scaled by 6.5 / (13.0 + 1e-6) ~= 0.5
    for (&val, &exp) in grads[0].to_vec().iter().zip(&[1.5, 2.0]) {
        assert!((val - exp).abs() < 1e-5);
    }
    for (&val, &exp) in grads[1].to_vec().iter().zip(&[0.0, 6.0]) {
        assert!((val - exp).abs() < 1e-5);
    }

    // No-op clipping when below threshold
    let unclipped_norm = clip_grad_norm_(&mut grads, 100.0, NormType::L2);
    assert!((unclipped_norm - 6.5).abs() < 1e-5);
}

// =============================================================================
// 6. StateDict Serialization Roundtrip
// =============================================================================

#[test]
fn test_state_dict_round_trip() {
    let mut sd = StateDict::new("Adam", 42);
    sd.insert_scalar("lr", 0.05);
    sd.insert_tensor("exp_avg.0", Tensor::from_slice(&[0.1, 0.2, 0.3], vec![3]));

    let bytes = sd.save_bytes();
    let loaded = StateDict::from_bytes(&bytes).unwrap();

    assert_eq!(loaded.metadata.optimizer_type, "Adam");
    assert_eq!(loaded.metadata.step, 42);
    assert_eq!(loaded.get_scalar("lr"), Some(0.05));
    let t = loaded.get_tensor("exp_avg.0").unwrap();
    assert_eq!(t.shape(), &[3]);
    assert_eq!(t.to_vec(), vec![0.1, 0.2, 0.3]);
}

// =============================================================================
// 7. Advanced Lion Optimizer & Schedulers
// =============================================================================

#[test]
fn test_lion_optimizer_step_trajectory() {
    use brain_optim::lion::{Lion, LionConfig};

    let mut params = vec![Tensor::from_slice(&[1.0, -1.0], vec![2])];
    let group = ParamGroup::new(vec![0], 0.1);
    let mut opt = Lion::new(
        vec![group],
        LionConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        },
    );

    let grad = Tensor::from_slice(&[0.5, -0.5], vec![2]);
    let info = opt.step(&mut params, &[grad]).unwrap();

    assert_eq!(info.step_count, 1);
    // On step 1: m_0 = 0 -> c = 0.1 * g -> sign(0.05)=1.0, sign(-0.05)=-1.0
    // theta_0 = 1.0 - 0.1 * 1.0 = 0.9
    // theta_1 = -1.0 - 0.1 * (-1.0) = -0.9
    let p = &params[0];
    assert!((p.get(0) - 0.9).abs() < 1e-6);
    assert!((p.get(1) - (-0.9)).abs() < 1e-6);
}

#[test]
fn test_onecycle_lr_schedule() {
    use brain_optim::schedulers::onecycle::{AnnealStrategy, OneCycleConfig, OneCycleLR};

    let mut opt = Sgd::new(vec![ParamGroup::new(vec![0], 0.01)], SgdConfig::default());
    let mut sched = OneCycleLR::new(
        vec![0.1],
        OneCycleConfig {
            max_lr: 0.1,
            total_steps: 10,
            pct_start: 0.3,
            anneal_strategy: AnnealStrategy::Cosine,
            div_factor: 10.0,
            final_div_factor: 100.0,
            three_phase: false,
        },
    );

    let initial_lr = sched.get_last_lr()[0];
    assert!((initial_lr - 0.01).abs() < 1e-6);

    let lrs_1 = sched.step(&mut opt).unwrap();
    assert!(lrs_1[0] > 0.0);
}
