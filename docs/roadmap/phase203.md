# Phase 203: ONNX Round-Trip Hardening for Interop

**Stage:** Post-1.0 Ecosystem
**Depends on:** Phase 202
**Status:** ✅ Complete in `brain-onnx`

## Objective
Extend ONNX import/export pipeline (opsets 9–21) to execute and validate external ONNX models published by PyTorch, TensorFlow, and ONNX Runtime.

## Deliverables
- Pure-Rust protobuf parser and execution interpreter in `crates/brain-onnx/`.
- Integration tests in `crates/brain-onnx/tests/onnx_roundtrip.rs`.
