#!/usr/bin/env bash
set -euo pipefail

echo "=== Preparing Brain 1.0 Release ==="

./scripts/ci.sh

echo "Building release binary..."
cargo build -p brain --release -j 2

echo "Running examples validation..."
cargo run --example convnet_train -j 2
cargo run --example onnx_export --features export -j 2
cargo run --example quantize_linear --features export -j 2

echo "Brain 1.0 is verified and ready for release tagging."
