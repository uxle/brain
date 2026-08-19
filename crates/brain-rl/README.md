# `brain-rl`

Pure-Rust reinforcement learning toolkit: DQN family, PPO, A2C, SAC, actor-critic, environments, and replay buffers.

## Overview

`brain-rl` provides a complete RL stack built on `brain-core` tensors, `brain-nn` layers, and `brain-autograd` — all in 100% safe Rust with zero external dependencies. It ships classic gym-style environments, tabular and neural value functions, exploration policies, and a suite of modern agents, plus a `DqnTrainer` for end-to-end training loops.

## Features

- **DQN family**: `DqnAgent`, `DoubleDqnAgent`, `DuelingDqnAgent`, `RainbowAgent` with configurable `DqnConfig` (gamma, LR, target-update frequency, epsilon decay).
- **Policy-gradient agents**: `PpoAgent` (with `PpoClippedObjective` clipped surrogate + value losses), `A2cAgent`, `SacAgent` (entropy-regularized), and a shared `ActorCriticNet` with `compute_gae`.
- **Environments**: `CartPoleEnv`, `MountainCarEnv`, `PendulumEnv`, `GridWorldEnv`, `CliffWalkingEnv`, `FrozenLakeEnv`, lite Atari (`PongLiteEnv`, `BreakoutLiteEnv`) and MuJoCo-style (`HalfCheetahLiteEnv`, `ReacherLiteEnv`) envs, `DummyVecEnv` vectorization, and `FrameStackWrapper`.
- **Buffers**: `ReplayBuffer`, `PrioritizedReplayBuffer` (with `SumTree`), `NStepBuffer`, and `TrajectoryBuffer`.
- **Policies & value functions**: `EpsilonGreedyPolicy` with `EpsilonSchedule`, `GaussianPolicy`, `CategoricalDist`, `DiagonalGaussianDist`; `QNet`/`VNet` and tabular `QTable`/`VTable`.
- **Tooling**: `DqnTrainer`/`TrainerConfig`, `evaluate_dqn`/`EvalReport`, `RlCheckpoint` (save/load DQN), `discount_returns`, `moving_average`, and a `prelude` module.

## Modules

| Module | Contents |
|---|---|
| `core` | `Space`, `Transition`, `Trajectory`, `RlError`/`RlResult` |
| `env` | `Env` trait, `EnvStep`, 10+ environments, vector & wrapper helpers |
| `policy` | `Policy`, exploration schedules, categorical/gaussian distributions |
| `value` | `ValueFn`, `QNet`, `QTable`, `VNet`, `VTable` |
| `buffer` | `ReplayBuffer`, `PrioritizedReplayBuffer`, `SumTree`, `NStepBuffer`, `TrajectoryBuffer` |
| `dqn`, `ppo`, `a2c`, `sac`, `actor_critic` | Agents, configs, clipped objectives, GAE |
| `agents` | `Agent` trait, `AgentKind`, `make_agent` factory |
| `trainer`, `eval`, `checkpoint`, `utils` | Training loop, evaluation, checkpointing, helpers |

## Quick Start

```rust
use brain_core::Tensor;
use brain_rl::{DqnAgent, DqnConfig, ReplayBuffer, Transition};

let mut agent = DqnAgent::new(4, 2, DqnConfig::default());
let mut buffer = ReplayBuffer::new(10_000);

let state = Tensor::zeros(vec![4]);
let action = agent.act(&state);
buffer.push(Transition::new(state, action, 1.0, Tensor::zeros(vec![4]), false));
```

## Testing

```bash
cargo test -p brain-rl -j 2
```

## Workspace Role

Depends on `brain-core` (tensors), `brain-nn` (network layers), `brain-optim` (optimizers), and `brain-autograd` (differentiation) to provide ready-made RL agents and training loops for the Brain framework.