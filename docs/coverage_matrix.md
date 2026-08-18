# Brain Deep Learning Framework — 1.0 Coverage Matrix

This matrix specifies the correctness, differentiability, and test coverage status for every core operation, layer, loss function, and optimizer across the Brain framework.

## 1. Core Operators & Tensors (`brain-core`)

| Operation | DType Support | Forward Verified? | Backward VJP? | Finite-Difference Checked? | Cache-Blocked GEMM? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `add`, `sub`, `mul`, `div` | F64, F32 | Yes | Analytical | Yes (`grad_check`) | N/A |
| `matmul` | F64, F32 | Yes (`numerical_check`) | Analytical | Yes (`grad_check`) | Yes (64x64 blocked) |
| `bmm`, `addmm` | F64, F32 | Yes | Analytical | Yes | Yes |
| `conv2d`, `conv2d_ext` | F64, F32 | Yes (Dilation verified) | Analytical | Yes (`grad_conv2d`) | N/A |
| `conv_transpose2d` | F64, F32 | Yes | Analytical | Yes (`grad_conv_transpose2d`) | N/A |
| `max_pool2d` | F64, F32 | Yes | Analytical | Yes (`grad_max_pool2d`) | N/A |
| `avg_pool2d`, `avg_pool2d_ext` | F64, F32 | Yes | Analytical | Yes (`grad_avg_pool2d`) | N/A |
| `relu`, `sigmoid`, `tanh` | F64, F32 | Yes | Analytical | Yes | N/A |
| `softmax`, `log_softmax` | F64, F32 | Yes (Numerically stable) | Analytical | Yes | N/A |
| `transpose`, `reshape`, `permute` | All | Yes | Analytical | Yes | N/A |

---

## 2. Neural Network Layers (`brain-nn` / `brain-train`)

| Layer | Value Differentiable? | Parameter Gradients? | Regularization? | Checkpoint Load/Save? |
| :--- | :--- | :--- | :--- | :--- |
| `Linear` | Yes (`Value::linear`) | Analytical (`dW = dY^T @ X`) | L1 / L2 | Yes (`ModelState`) |
| `Conv2d` | Yes (`Value::conv2d`) | Analytical (`grad_conv2d`) | L1 / L2 | Yes (`ModelState`) |
| `ConvTranspose2d` | Yes (`Value::conv_transpose2d`) | Analytical | L1 / L2 | Yes (`ModelState`) |
| `MaxPool2d` | Yes (`Value::max_pool2d`) | Mask-based Argmax VJP | None | N/A |
| `AvgPool2d` | Yes (`Value::avg_pool2d`) | Uniform spread VJP | None | N/A |
| `Flatten` | Yes (`Value::reshape`) | Shape-restoration VJP | None | N/A |
| `Embedding` | Yes (`Value::embedding`) | Index-scatter gradient | L2 | Yes (`ModelState`) |
| `BatchNorm2d` | Yes (Running stats tracking) | Mean / Var normalization | None | Yes |

---

## 3. Loss Functions (`brain-loss`)

| Loss Function | Classification / Regression | `forward_value` Path? | Mathematical Reference Checked? |
| :--- | :--- | :--- | :--- |
| `CrossEntropyLoss` | Classification | Yes (Log-softmax + target select) | Analytical vs Central Finite Diff |
| `BinaryCrossEntropyLoss` | Classification | Yes | Analytical vs Central Finite Diff |
| `MSELoss` | Regression | Yes | Analytical vs Central Finite Diff |
| `SmoothL1Loss` | Regression | Yes (Huber-style quadratic/linear) | Analytical vs Central Finite Diff |

---

## 4. Optimizers (`brain-optim`)

| Optimizer | Closed-Form 1-Step Validated? | Decoupled Weight Decay? | State Serialization (`save_bytes`)? |
| :--- | :--- | :--- | :--- |
| `SGD` | Yes ($f(x)=x^2 \to 0.8$) | Supported | Bit-identical roundtrip |
| `Adam` | Yes (Bias-corrected 1st step $\to 0.9$) | $L_2$ penalty | Bit-identical roundtrip |
| `AdamW` | Yes ($-\text{lr} \cdot \lambda \cdot x \to 0.899$) | Decoupled | Bit-identical roundtrip |
| `StepLR`, `CosineAnnealingLR` | Yes (Validated schedule step updates) | N/A | State preserved |

---

## 5. Compilers, Distributed & Mixed Precision

| Component | Crate | Verification Target | Status |
| :--- | :--- | :--- | :--- |
| `IrGraph` & `Interpreter` | `brain-compile` | `tests/compile_equivalence.rs` | Equivalence verified ($<1e-6$) |
| `DataParallel` & `AllReduce` | `brain-distributed` | `tests/distributed_equiv.rs` | Deterministic 2-rank mock verified |
| `GradScaler` (AMP) | `brain-autograd` | `tests/amp_scaler_test.rs` | Dynamic loss scale, unscale & backoff verified |

---

## 6. Ecosystem & Quantization

| Component | Crate | Target | Status |
| :--- | :--- | :--- | :--- |
| `OnnxModel` & `evaluate_onnx_model` | `brain-onnx` | `tests/onnx_roundtrip.rs` | Validated opset 17 |
| `quantize_tensor` (Int8) | `brain-quantization` | `tests/quant_linear.rs` | Verified ($<1e-2$) vs fp32 |
| `apply_magnitude_prune` | `brain-quantization` | `tests/quant_linear.rs` | 50% sparsity verified |
