# `brain-export` (v0.2.0)

> Multi-Format Model Interoperability: Standalone Zero-Dependency Exporters for ONNX, TFLite, CoreML, and WebNN.

## Overview

`brain-export` enables exporting Brain models to industry-standard runtime formats with zero external dependencies (no flatc, no protobuf compilers). It writes valid binary Protobuf ONNX files, binary FlatBuffers TFLite models, Apple CoreML `.mlpackage` archives, and W3C WebNN computational graph representations.

## Architecture

| Module | Description |
|---|---|
| `onnx` | Standalone binary ONNX Protobuf serializer, operator registry, and graph validator |
| `tflite` | Standalone binary FlatBuffers schema builder for TensorFlow Lite models |
| `coreml` | Apple CoreML model package builder and neural network layer generator |
| `webnn` | W3C WebNN JSON graph specification exporter and `MLGraphBuilder` generator |
| `verify` | Numerical tolerance export verifier comparing reference outputs against exported graphs |
| `zip` | Pure Rust ZIP archive packer for `.mlpackage` and zipped export bundles |

## Quality & Verification

- **Tests**: 13,513 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-export -- -D warnings`)
- **Dependencies**: `std` + `brain-core`
