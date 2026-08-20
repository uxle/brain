//! Comprehensive Reinforcement Learning Algorithms Test Suite

use brain_core::Tensor;
use brain_rl::*;

#[test]
fn test_gae_computation_correctness() {
    let rewards = vec![1.0, 1.0, 1.0];
    let values = vec![0.5, 0.5, 0.5];
    let dones = vec![false, false, true];
    let next_val = 0.0;
    let gamma = 0.99;
    let lambda = 0.95;

    let (advantages, returns) = compute_gae(&rewards, &values, &dones, next_val, gamma, lambda);
    assert_eq!(advantages.len(), 3);
    assert_eq!(returns.len(), 3);

    // Terminal step delta_2 = r_2 - V_2 = 1.0 - 0.5 = 0.5
    assert!((advantages[2] - 0.5).abs() < 1e-6);
    assert!((returns[2] - 1.0).abs() < 1e-6);
}

#[test]
fn test_ppo_clipped_surrogate_loss() {
    let ppo = PpoClippedObjective::new(0.2); // epsilon = 0.2

    // Positive advantage A > 0: loss is bounded when ratio > 1.2
    let loss_normal = ppo.compute_policy_loss(1.1, 2.0); // -2.2
    let loss_clipped = ppo.compute_policy_loss(1.5, 2.0); // -1.2 * 2.0 = -2.4
    assert!((loss_normal - (-2.2)).abs() < 1e-6);
    assert!((loss_clipped - (-2.4)).abs() < 1e-6);

    // Value loss clipping
    let v_loss = ppo.compute_value_loss(1.0, 0.5, 2.0);
    assert!(v_loss > 0.0);
}

#[test]
fn test_prioritized_replay_buffer_insert_and_sample() {
    let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
    let s = Tensor::from_slice(&[1.0, 0.0], vec![2]);
    let s_next = Tensor::from_slice(&[0.0, 1.0], vec![2]);

    for i in 0..5 {
        let t = Transition::new(s.clone(), i, 1.0 + i as f64, s_next.clone(), false);
        per.push(t);
    }

    assert_eq!(per.buffer.len(), 5);
    let (indices, transitions, weights) = per.sample_batch(3).unwrap();
    assert_eq!(transitions.len(), 3);
    assert_eq!(indices.len(), 3);
    assert_eq!(weights.len(), 3);
}

#[test]
fn test_dqn_agent_step_and_eval() {
    let cfg = DqnConfig::default();
    let mut agent = DqnAgent::new(4, 2, cfg);
    let state = Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4], vec![4]);
    let action = agent.act(&state);
    assert!(action < 2);
}

#[test]
fn test_sac_config_and_creation() {
    let cfg = SacConfig::default();
    let agent = SacAgent::new(3, 1, cfg);
    assert_eq!(agent.config.alpha, 0.2);
}

#[test]
fn test_cartpole_and_gridworld_env() {
    let mut cp = CartPoleEnv::new();
    let s0 = cp.reset().unwrap();
    assert_eq!(s0.shape(), &[4]);
    let step = cp.step(1).unwrap();
    assert_eq!(step.observation.shape(), &[4]);

    let mut gw = GridWorldEnv::new(4, 4);
    let g0 = gw.reset().unwrap();
    assert_eq!(g0.shape(), &[2]);
    let g_step = gw.step(0).unwrap();
    assert_eq!(g_step.observation.shape(), &[2]);
}
