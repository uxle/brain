# Phase 0: Unify `brain-nn` onto the `brain-autograd::Value` Tape

**Priority:** Core Architectural Invariant.
**Depends on:** `brain-autograd::Value` generic reverse-mode automatic differentiation tape.
**Blocks:** True end-to-end differentiable Python bindings (`brain-python`), unified GPU compilation (`brain-compile`), and eliminates redundant duplicate gradient formulas in `brain-train`.

---

## 1. Problem Statement & Architectural Alignment

Previously, three disconnected backward systems coexisted:
1. `brain-autograd::Value`: A true dynamic tape supporting scalar/vector operations, activations, convolutions, and backward traversal.
2. `brain-nn`: Contained 75 forward-only layer structures operating on raw `Tensor` without tape graph registration.
3. `brain-train`: Manually maintained hand-derived analytical gradient dispatchers for a small subset of layers.

By unifying `brain-nn::Module` to operate on `brain_autograd::Value`:
- Every layer in `brain-nn` (`Linear`, `Conv2d`, `ConvTranspose2d`, `MaxPool2d`, `AvgPool2d`, `Embedding`, `Sequential`, `LayerNorm`, `BatchNorm2d`, `LSTM`, `MultiheadAttention`) automatically builds a computation graph on the tape.
- Calling `.backward()` on any loss computed from layer outputs computes exact analytical gradients through the unified vector-Jacobian products in `brain-autograd`.
- `brain-train`, `brain-python`, `brain-rl`, `brain-vit`, `brain-transformer`, and `brain-gnn` share a single source of truth for differentiation.

---

## 2. Implementation Deliverables

1. **`Module` Trait Evolution (`crates/brain-nn/src/module/mod.rs`)**:
   - `forward_value(&self, input: &Value) -> ModuleResult<Value>`
   - `parameters_value(&self) -> Vec<Value>`
   - Retained `forward(&self, input: &Tensor)` transitional shim for zero-breakage downstream compatibility.

2. **Tier 1 Layer Migration**:
   - `Linear` (`crates/brain-nn/src/layers/linear.rs`): Direct `Value::linear` tape registration.
   - `Conv2d` (`crates/brain-nn/src/layers/conv.rs`): Direct `Value::conv2d` spatial convolution graph building.
   - `ConvTranspose2d` (`crates/brain-nn/src/layers/conv_transpose.rs`): Direct `Value::conv_transpose2d` deconvolution.
   - `MaxPool2d` & `AvgPool2d` (`crates/brain-nn/src/layers/pool.rs`): Dynamic pooling VJPs.
   - `Embedding` (`crates/brain-nn/src/layers/embedding.rs`): Dynamic index lookup and gradient accumulation.
   - `Sequential` (`crates/brain-nn/src/containers/seq.rs`): Sequential chaining of `Value` nodes.
   - `ReLU` (`crates/brain-nn/src/activations/relu.rs`): Nonlinear activation graph propagation.

3. **Integration Verification**:
   - Created `crates/brain-nn/tests/unified_autograd_tape_test.rs` asserting real `.backward()` passes on `Linear`, `Conv2d`, `ConvTranspose2d`, `Embedding`, and multi-layer `Sequential` models.

---

## 3. Verification Protocol

```bash
cargo test -p brain-nn --test unified_autograd_tape_test -j 2
cargo test -p brain-nn --test layer_grad_check -j 2
./scripts/ci.sh
```
