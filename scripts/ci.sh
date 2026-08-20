#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Brain 1.0 Local CI Suite (Bounded Jobs: -j 2) ==="

echo "1. Checking Core Tensor Numerics..."
cargo test -p brain-core --test numerical_check -j 2

echo "2. Checking Autograd Gradients & Tape Bounds..."
cargo test -p brain-autograd --test grad_check --test tape_memory_bounded -j 2

echo "2b. Checking Neural Network Layers & Unified Tape..."
cargo test -p brain-nn -j 2

echo "3. Checking Losses..."
cargo test -p brain-loss -j 2

echo "4. Checking Optimizers..."
cargo test -p brain-optim --test optim_step_test -j 2

echo "5. Checking Trainer Regressions..."
cargo test -p brain-train --test trainer_regression -j 2

echo "6. Checking ONNX Roundtrip..."
cargo test -p brain-onnx --test onnx_roundtrip -j 2

echo "7. Checking Quantization..."
cargo test -p brain-quantization --test quant_linear -j 2

echo "8. Checking CLI..."
cargo test -p brain-cli -j 2

echo "9. Checking Python Bindings & PyO3 API..."
cargo build -p brain-python -j 2
cp target/debug/libbrain_native.so python/brain/brain_native.so
PYTHONPATH=python python3 -m unittest discover -s tests/python

echo "=== All Tests Passed Cleanly ==="
