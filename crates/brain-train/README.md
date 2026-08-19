# `brain-train`

Integrated training loop: trainable layers, sequential models, trainer, callbacks, and serializable model state.

## Overview

`brain-train` provides the workspace's first mutable model surface: `brain-nn::Module` layers expose cloned parameters, so this crate supplies trainable `Linear`, `Conv2d`, pooling and flatten layers whose parameter tensors can be updated in place by `brain-optim` optimizers after loss gradients are computed. It bundles a `Trainer` with mini-batching, gradient accumulation, callbacks, and a text-based `ModelState` checkpoint format.

## Features

- **Trainable layers** — `Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Flatten`, `ReLU`, assembled in a `Sequential` via `add`.
- **`TrainableModule` trait** — `forward`, `parameters`, `load_parameters`, `parameter_names` over mutable parameter state.
- **`Trainer`** — builder-driven (`model`, `loss`, `optimizer`, `learning_rate`, `regularizer`), with `train_batch`, `fit`, `fit_accumulated` (gradient accumulation), `evaluate`, and `summary`.
- **Data helpers** — `Batch` (`[batch, features]` inputs + integer targets) and `SyntheticClassification` with deterministic mini-batches.
- **Regularization** — `Regularizer` trait and built-in `L2Regularization` hook.
- **Checkpointing** — `ModelState` / `NamedTensor` with `to_brain_bytes` / `from_brain_bytes` text format and `to_bytes` / `from_bytes` aliases.
- **Callbacks** — `TrainingCallback` trait with `EarlyStopping` and `MetricHistoryLogger` (`CallbackAction::Stop`/`Continue`).
- **Adapters** — `TensorModuleAdapter` wraps tensor-only `Module`s; `tensor_to_value` / `value_to_tensor` bridge `Tensor` ↔ `Value`.

## Modules

| Module | Description |
|---|---|
| `lib` | `Trainer`, `TrainerBuilder`, `Sequential`, trainable layers, `Batch`, `ModelState`, `SyntheticClassification` |
| `callbacks` | `TrainingCallback`, `EarlyStopping`, `MetricHistoryLogger`, `CallbackAction` |

## Quick Start

```rust
use brain_train::{Linear, ReLU, Sequential, SyntheticClassification, Trainer};

let data = SyntheticClassification::two_class_points(8);
let batches = data.batches(4);
let model = Sequential::new()
    .add(Linear::new(2, 4, true))
    .add(ReLU::new())
    .add(Linear::new(4, 2, true));

let mut trainer = Trainer::builder()
    .model(model)
    .learning_rate(0.2)
    .build()
    .unwrap();
let summary = trainer.fit(&batches, 8).unwrap();
println!("accuracy = {}", summary.accuracy);
```

## Testing

```bash
cargo test -p brain-train -j 2
```

Integration tests train a synthetic MLP and CNN end-to-end and round-trip `ModelState` checkpoints.

## Workspace Role

Depends on `brain-core`, `brain-autograd`, `brain-loss`, `brain-metric`, `brain-nn`, and `brain-optim`. Consumers: `brain-cli` and the `brain` facade (via its `train` feature).
