# `brain`

Feature-gated facade crate and CLI binary for the Brain deep learning framework.

## Overview

`brain` is the workspace's user-facing entry point. As a library it re-exports the stable, wired parts of the workspace behind Cargo features — `core`, `autograd`, `nn`, `loss`, `optim`, `train`, `data`, `metric`, and the `export` group (`graph`, `onnx`, `quantization`, `artifact`) — so applications can depend on one crate instead of many. As a binary it forwards all arguments to `brain-cli`'s `run_cli` dispatcher.

## Features

- **Facade re-exports** — `brain::core`, `brain::autograd`, `brain::nn`, `brain::loss`, `brain::optim`, `brain::train`, `brain::data`, `brain::metric`, `brain::export::{graph, onnx, quantization, artifact}`.
- **Feature flags** — `train` (default), `cli` (default), `autograd`, `data`, `export`, `loss`, `metric`, `nn`, `optim`; default = `["train", "cli"]`.
- **CLI binary** — thin `main` that forwards args to `brain_cli::run_cli(&args, &OutputSink::stdout())` and exits with the returned code.
- **Examples** — `tiny_mlp_train` (end-to-end training with `brain-train`) and `export_onnx` (ONNX export pipeline).
- **Smoke tests** — `framework_smoke` verifies the facade wiring across enabled features.

## Quick Start

```rust
use brain::core::Tensor;
use brain::core::tensor::arithmetic::matmul;

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
let c = matmul(&a, &b);
println!("{:?}", c.to_vec());
```

From a shell:

```bash
brain make my-project     # scaffold a project
brain train config.json   # train a model
brain check model.onnx    # validate an ONNX model
brain repl                # interactive session
```

## Testing

```bash
cargo test -p brain -j 2
```

(The `-j 2` cap avoids CPU/memory starvation in this 33-crate workspace.)

## Workspace Role

Depends on `brain-core` (always) plus optional `brain-autograd`, `brain-cli`, `brain-data`, `brain-dataset`, `brain-export`, `brain-graph`, `brain-loss`, `brain-metric`, `brain-nn`, `brain-optim`, `brain-train`, `brain-onnx`, and `brain-quantization` behind features. It is the top-level crate: nothing depends on it.