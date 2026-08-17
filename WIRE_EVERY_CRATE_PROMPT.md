# Prompt: Wire Every Brain Crate Into One Coherent Deep Learning Framework

You are working in the Rust workspace at `/home/lion/Documents/GitHub/brain`.

Your mission is to turn Brain from a collection of individually compiling crates into one coherent, end-to-end deep learning framework. Be ambitious about the product quality, but be brutally honest in the implementation: no fake claims, no placeholder integrations presented as complete, and no README inflation. The result should compile, have real cross-crate examples, and expose a clean developer experience.

## Current Project Shape

The workspace currently contains these crates:

- Foundation: `brain-core`, `brain-autograd`, `brain-nn`, `brain-optim`, `brain-loss`, `brain-metric`, `brain-regularization`, `brain-utils`
- Model families: `brain-transformer`, `brain-rnn`, `brain-vit`, `brain-cv`, `brain-text`, `brain-audio`, `brain-gnn`, `brain-gan`, `brain-diffusion`, `brain-rl`, `brain-neuroevolution`
- Systems and deployment: `brain-graph`, `brain-compile`, `brain-quantization`, `brain-onnx`, `brain-export`, `brain-distributed`, `brain-federated`, `brain-data`, `brain-dataset`, `brain-benchmark`, `brain-cli`

Important observations from the repo:

- `cargo check` currently passes.
- Most crates depend only on `brain-core`; cross-crate wiring is thin.
- A few crates are partially linked: `brain-rl` depends on `brain-nn`, `brain-optim`, and `brain-autograd`; `brain-rnn` and `brain-text` depend on `brain-autograd`; `brain-onnx` and `brain-quantization` depend on `brain-graph`.
- The central abstractions are currently split across crates:
  - `brain-core::Tensor`
  - `brain-autograd::Value`
  - `brain-nn::Module`
  - `brain-optim::Optimizer`
  - `brain-loss::Loss`
  - `brain-metric::Metric`
- `brain-nn::Module::forward` currently takes `&Tensor` and returns `Tensor`, while autograd operates around `Value`. This is the main training-loop integration gap.
- There are no obvious top-level `examples`, `tests`, or `benches` directories, so the framework lacks visible end-to-end proof that the crates work together.
- The README makes very large claims about completeness and test counts. Keep documentation aligned with what the code actually proves.

## High-Level Goal

Create the first truly unified Brain framework surface:

1. A user can build a model with `brain-nn`, `brain-transformer`, `brain-rnn`, `brain-vit`, or domain crates.
2. Inputs come from `brain-data` / `brain-dataset`.
3. Forward computation can participate in autograd.
4. Losses from `brain-loss` can produce trainable scalar objectives.
5. Optimizers from `brain-optim` can update model parameters.
6. Metrics from `brain-metric` can evaluate predictions.
7. Regularization hooks from `brain-regularization` can compose with training.
8. Graph/export/quantization/compile crates can consume trained model surfaces through explicit adapters.
9. Distributed/federated crates integrate through stable training state and parameter APIs.
10. The CLI can run at least one real training, evaluation, export, and benchmark workflow.

## Architecture Rules

- Do not simply add every crate as a dependency of every other crate. Design a layered dependency graph.
- Keep `brain-core` minimal and dependency-free.
- Put cross-cutting public traits where they naturally belong, then add adapter crates or feature-gated integrations where needed.
- Avoid dependency cycles. If two crates need each other, extract the shared trait/API into a lower-level crate.
- Prefer feature flags for optional integrations such as `onnx`, `distributed`, `quantization`, `rl`, `vision`, `text`, and `audio`.
- Keep public APIs small, typed, and documented.
- Do not use unsafe code unless absolutely necessary and explicitly justified.
- Preserve the project’s current preference for pure Rust and std-only internals unless a crate already uses or explicitly accepts an external dependency.

## Concrete Wiring Plan

### 1. Create a Unified Prelude

Add a top-level ergonomic import path, either by introducing a `brain` facade crate or strengthening `brain-cli`/workspace-level docs with a recommended app crate pattern.

The ideal user API should feel like:

```rust
use brain::prelude::*;

let model = Sequential::new()
    .add(Linear::new(784, 256, true))
    .add(ReLU::new())
    .add(Linear::new(256, 10, true));

let trainer = Trainer::builder()
    .model(model)
    .loss(CrossEntropyLoss::default())
    .optimizer(AdamW::default())
    .metric(Accuracy::default())
    .build();
```

If a `brain` facade crate does not exist, create it as a new workspace member that re-exports stable APIs from the other crates behind features.

### 2. Bridge `Tensor`, `Value`, `Module`, and `Optimizer`

Design and implement the missing training bridge:

- Add an autograd-aware module trait such as `AutogradModule`, `TrainableModule`, or `DifferentiableModule`.
- Provide adapters from `brain-nn::Module` to autograd where possible.
- Make parameters track gradients in a consistent representation.
- Decide whether trainable parameters should be stored as `Tensor`, `Value`, or a wrapper type.
- Ensure optimizer `step` can consume gradients produced by autograd without manual glue in user code.
- Add clear conversions:
  - `Tensor -> Value`
  - `Value -> Tensor`
  - model parameters -> optimizer param groups
  - model state -> checkpoint/export state

### 3. Build a Real Training Loop

Introduce a shared trainer API, likely in a new `brain-train` crate or a carefully chosen existing crate.

The trainer should integrate:

- `brain-data` / `brain-dataset` batches
- `brain-nn::Module` or autograd-aware model traits
- `brain-loss::Loss`
- `brain-optim::Optimizer`
- `brain-metric::Metric`
- `brain-regularization` hooks
- checkpoint state
- evaluation mode
- deterministic seeding

Add at least one real example that trains a tiny MLP on synthetic classification data.

### 4. Wire Domain Model Crates Into Core Training

Add adapters and examples for:

- `brain-transformer` + `brain-text`: tokenize text, build embeddings, run transformer, compute language-model loss.
- `brain-vit` + `brain-cv`: image batch, patch embedding, classification head, metric evaluation.
- `brain-rnn` + sequence data: packed or padded sequence training.
- `brain-rl` + `brain-nn` + `brain-optim`: agent update loop using shared optimizer and checkpoint APIs.
- `brain-gan` / `brain-diffusion`: generator training loops using shared loss, optimizer, metrics, and export surfaces.
- `brain-gnn`: graph batch training using shared trainer concepts where practical.

Keep these feature-gated so users do not pay compile-time or dependency costs for domains they do not use.

### 5. Wire Graph, Compile, Quantization, ONNX, and Export

Create explicit model-to-graph and graph-to-export paths:

- `brain-nn` / transformer / vit / rnn models should lower into `brain-graph` IR where supported.
- `brain-graph` should feed `brain-compile`.
- `brain-graph` should feed `brain-onnx`.
- `brain-quantization` should operate on tensors and graph IR, with examples for post-training quantization and quantization-aware training where possible.
- `brain-export` should become the user-facing export layer and call `brain-onnx`, TFLite/CoreML/WebNN paths as appropriate.

Add one proof example:

```text
train tiny MLP -> lower to graph -> optimize graph -> export ONNX bytes -> validate with brain-onnx checker
```

### 6. Wire Distributed and Federated Training

Expose training state in a way that `brain-distributed` and `brain-federated` can consume:

- model parameter state
- optimizer state
- gradients
- metrics
- checkpoints
- round/epoch summaries

Add minimal examples:

- data-parallel local simulation with 2 workers
- federated averaging over synthetic clients

### 7. Make the CLI a Real Product Surface

The CLI should support at least:

- `brain train examples/mlp.toml`
- `brain eval checkpoints/mlp.brain`
- `brain export checkpoints/mlp.brain --format onnx`
- `brain benchmark --suite tiny`
- `brain doctor`

If the current `brain-cli` is library-only, add a binary target or document how it is invoked.

### 8. Add Examples and Integration Tests

Add top-level directories:

- `examples/`
- `tests/`
- optionally `benches/`

Minimum examples:

- `examples/tiny_mlp_train.rs`
- `examples/text_transformer_lm.rs`
- `examples/vit_image_classifier.rs`
- `examples/export_onnx.rs`
- `examples/quantize_model.rs`
- `examples/rl_cartpole.rs`

Minimum integration tests:

- Tensor + autograd scalar backward.
- Module + loss + optimizer one-step training.
- Dataset + trainer smoke test.
- Metric accumulation during evaluation.
- Model state save/load round trip.
- Graph lowering smoke test.
- ONNX export validation smoke test.
- Quantization smoke test.
- CLI `doctor` smoke test.

### 9. Documentation Must Match Reality

Update README files to describe what is actually complete.

Include:

- crate dependency diagram
- recommended feature flags
- quick start using the unified facade
- training example
- export example
- current limitations
- roadmap

Remove or qualify unproven claims like “surpasses PyTorch and TensorFlow” unless backed by real benchmarks and documented tradeoffs.

## Acceptance Criteria

The work is complete only when all of these pass:

```bash
cargo fmt --all --check
cargo check --workspace --all-targets --all-features
cargo test --workspace --all-targets
```

Also run targeted example checks:

```bash
cargo run --example tiny_mlp_train
cargo run --example export_onnx
```

If an example is too expensive to run by default, provide a tiny deterministic mode that finishes quickly.

## Final Deliverable

Deliver:

- A concise architectural summary of the final crate dependency graph.
- A list of public APIs added or changed.
- The examples/tests added.
- Exact verification commands and results.
- Any remaining limitations or honest next steps.

Remember: the goal is not just “every crate depends on every crate.” The goal is that a serious Rust developer can use Brain as one framework: data in, model forward, autograd backward, optimizer step, metrics, checkpoint, export, quantize, compile, distribute, and ship.
