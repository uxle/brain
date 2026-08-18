# Brain 1.0 Public API Surface

This document provides a concise reference to the primary public traits, structs, and functions exposed by the core Brain 1.0 crates.

## `brain-core`
- **Tensor Types**: `Tensor`, `Shape`, `Dim`, `Strides`, `DType`, `Device`
- **Arithmetic & BLAS**: `matmul`, `bmm`, `addmm`, `dot`, `conv2d`, `conv2d_ext`, `conv_transpose2d`, `max_pool2d`, `avg_pool2d`
- **Transforms & Reductions**: `transpose`, `reshape`, `permute`, `softmax`, `log_softmax`, `sum`, `mean`, `max`

## `brain-autograd`
- **Core Types**: `Value`, `GradFn`, `Tape`, `OpRecord`
- **Operations on Value**: `linear`, `conv2d`, `conv_transpose2d`, `max_pool2d`, `avg_pool2d`, `embedding`, `relu`, `sigmoid`, `tanh`, `sum`, `mean`, `transpose`, `reshape`, `log_softmax`
- **Engine**: `Value::backward(&self)`, `Value::grad(&self)`, `Tape::drain(&mut self)`

## `brain-nn`
- **Modules**: `Linear`, `Conv2d`, `ConvTranspose2d`, `Embedding`, `BatchNorm2d`, `LayerNorm`, `Dropout`, `Sequential`
- **Traits**: `Module`, `TrainableModule`, `ModuleMut`

## `brain-loss`
- **Loss Traits**: `Loss`, `ClassificationLoss`, `RegressionLoss`
- **Loss Types**: `CrossEntropyLoss`, `BinaryCrossEntropyLoss`, `MSELoss`, `SmoothL1Loss`

## `brain-optim`
- **Optimizers**: `Sgd`, `Adam`, `AdamW`
- **Schedulers**: `StepLR`, `CosineAnnealingLR`, `ExponentialLR`
- **Checkpointing**: `StateDict` (`save_bytes`, `from_bytes`)

## `brain-train`
- **Trainer Pipeline**: `Trainer`, `TrainerBuilder`, `Batch`, `ModelState`, `Sequential`, `Layer`
- **Training Methods**: `fit`, `fit_accumulated`, `train_batch`, `evaluate`, `load_state`, `state`

## `brain-onnx`
- **ONNX IR**: `OnnxModel`, `OnnxGraph`, `OnnxNode`, `OnnxValue`
- **Tools & Evaluation**: `evaluate_onnx_model`, `check_model`, `export_onnx_bytes`

## `brain-quantization`
- **Dynamic 8-Bit**: `quantize_tensor`, `dequantize_tensor`, `QuantConfig`, `QuantDType`, `QuantTensor`
- **Pruning**: `apply_magnitude_prune`, `MagnitudePruner`

## `brain-cli` / `brain`
- **CLI Commands**: `brain make`, `brain check`, `brain run`, `brain train`, `brain tensor`, `brain bench`, `brain dataset`, `brain doctor`
