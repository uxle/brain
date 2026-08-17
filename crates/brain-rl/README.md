# `brain-rl`

Production-grade Reinforcement Learning toolkit for the Brain deep learning framework.

## Overview

`brain-rl` provides a comprehensive, high-performance reinforcement learning ecosystem containing:
- **DQN Family**: Standard Deep Q-Networks (DQN), Double DQN (preventing value overestimation), Dueling DQN (state value $V(s)$ and advantage $A(s, a)$ stream separation), and Rainbow-Lite.
- **Policy Gradient & Actor-Critic**: Proximal Policy Optimization (PPO with clipped surrogate objective and value clipping), Synchronous Advantage Actor-Critic (A2C), and Soft Actor-Critic (SAC with maximum entropy objective).
- **Advantage Estimation**: Generalized Advantage Estimation ($\text{GAE}(\gamma, \lambda)$) and recursive return discounting.
- **Environment Suite**: Classic control physics (`CartPole`, `MountainCar`, `Pendulum`), Grid worlds (`GridWorld`, `CliffWalking`, `FrozenLake`), Atari-lite (`PongLite`, `BreakoutLite`), and continuous physics (`HalfCheetahLite`, `ReacherLite`).
- **Vectorized Environments & Wrappers**: Synchronous parallel stepping (`VecEnv`, `DummyVecEnv`), `FrameStack`, `TimeLimit`, `RewardScale`.
- **Replay Buffers**: Uniform cyclic ring buffer, Prioritized Experience Replay (`PrioritizedReplayBuffer` with binary `SumTree`), and `NStepBuffer` / `TrajectoryBuffer`.
- **Exploration Policies**: Epsilon-Greedy policies with linear/exponential annealing schedules, Categorical distributions, and Continuous Diagonal Gaussian policies.
- **Trainers, Evaluation & Checkpointing**: Integrated `DqnTrainer`, deterministic `evaluate_dqn`, and serialized `RlCheckpoint`.

---

## Architecture & Modules

```
brain-rl/
├── core.rs                 # Space, Transition, Trajectory, RlError, RlResult
├── env/
│   ├── mod.rs              # Env trait, EnvStep, EnvInfo
│   ├── gym.rs              # CartPole-v1, MountainCar-v0, Pendulum-v1 hand-written physics
│   ├── gridworld.rs        # GridWorld, CliffWalking, FrozenLake tabular benchmarks
│   ├── atari_lite.rs       # PongLite, BreakoutLite environments
│   ├── mujoco_lite.rs      # HalfCheetahLite, ReacherLite continuous planar physics
│   ├── wrappers.rs         # FrameStack, TimeLimit, RewardScale wrappers
│   └── vector.rs           # DummyVecEnv synchronous parallel execution
├── policy/
│   ├── mod.rs              # Policy trait, EpsilonGreedyPolicy, EpsilonSchedule
│   ├── dist.rs             # CategoricalDist, DiagonalGaussianDist, entropy, log_prob
│   └── gaussian.rs         # GaussianPolicy continuous action output
├── value/
│   ├── mod.rs              # ValueFn trait, VTable, VNet, target network soft update
│   └── qvalue.rs           # QTable tabular Q-learning, QNet linear Q-network
├── buffer/
│   ├── mod.rs              # ReplayBuffer cyclic ring buffer, BufferStats
│   ├── prioritized.rs      # SumTree binary tree, PrioritizedReplayBuffer (PER)
│   └── sequence.rs         # NStepBuffer n-step discounting, TrajectoryBuffer
├── dqn/
│   ├── mod.rs              # DqnAgent, DqnConfig, Huber loss update
│   ├── double.rs           # DoubleDqnAgent decoupled online/target action evaluation
│   ├── dueling.rs          # DuelingQNet, DuelingDqnAgent V(s) + A(s, a)
│   └── rainbow.rs          # RainbowAgent (Double + Dueling + PER)
├── ppo/
│   ├── mod.rs              # PpoAgent, PpoConfig, trajectory rollouts
│   └── clipped.rs          # PpoClippedObjective surrogate ratio clipping
├── a2c/
│   └── mod.rs              # A2cAgent synchronous advantage actor-critic
├── actor_critic/
│   └── mod.rs              # ActorCriticNet, Generalized Advantage Estimation (GAE)
├── sac/
│   └── mod.rs              # SacAgent, SacConfig, maximum entropy temperature tuning
├── agents/
│   └── mod.rs              # Agent trait, AgentKind, make_agent factory
├── trainer.rs              # DqnTrainer episode iterations & progress logging
├── eval.rs                 # evaluate_dqn deterministic evaluation & EvalReport
├── checkpoint.rs           # RlCheckpoint weight & replay serialization
├── utils.rs                # discount_returns, moving_average
└── lib.rs                  # Crate root, re-exports, prelude
```

---

## Quick Start

### 1. Training DQN on Classic CartPole

```rust
use brain_rl::prelude::*;

fn main() -> RlResult<()> {
    let mut env = CartPoleEnv::new();
    let config = DqnConfig {
        gamma: 0.99,
        lr: 1e-3,
        batch_size: 32,
        target_update_freq: 100,
        buffer_capacity: 10_000,
        epsilon_start: 1.0,
        epsilon_end: 0.05,
        epsilon_decay_steps: 1_000,
    };
    let mut agent = DqnAgent::new(4, 2, config);

    let mut state = env.reset()?;
    for step in 0..500 {
        let action = agent.act(&state);
        let step_res = env.step(action)?;

        let transition = Transition::new(
            state,
            action,
            step_res.reward,
            step_res.observation.clone(),
            step_res.done || step_res.truncated,
        );

        let _loss = agent.step(transition)?;
        state = step_res.observation;

        if step_res.done || step_res.truncated {
            state = env.reset()?;
        }
    }

    Ok(())
}
```

### 2. Prioritized Experience Replay (PER)

```rust
use brain_rl::buffer::PrioritizedReplayBuffer;
use brain_rl::core::Transition;
use brain_core::Tensor;

let mut per = PrioritizedReplayBuffer::new(10_000, 0.6, 0.4);
let s = Tensor::from_slice(&[1.0, 0.0], vec![2]);
let ns = Tensor::from_slice(&[0.9, 0.1], vec![2]);

per.push(Transition::new(s, 1, 1.0, ns, false));
let (tree_indices, batch, is_weights) = per.sample_batch(32)?;
```

---

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
