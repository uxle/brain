# `brain-export`

> Multi-format model export: ONNX, TFLite, CoreML, and WebNN targets from a single pure-Rust model abstraction.

## Overview

`brain-export` converts Brain models into industry-standard deployment formats without pulling in protobuf or FlatBuffers toolchains. It defines a shared `ExportModel` parameter container and a `ModelExporter` trait, with per-format configs, op-name mapping tables, a pure-Rust ZIP packer, and a numerical verification hook. The `.brain` binary container with CRC tamper detection lives in `brain-core`; this crate provides the format layer on top of it.

## Features

- `ModelExporter` trait with `OnnxExporter`, `TfliteExporter`, `CoreMlExporter`, `WebnnExporter` implementations and per-format configs (`OnnxConfig`, `TfliteConfig`, `CoreMlConfig`, `WebnnConfig`)
- Operator-name mapping to ONNX ops, TFLite builtin codes, CoreML layer checks, and WebNN ops (`map_to_onnx_op`, `map_to_tflite_builtin_code`, `is_coreml_layer_supported`, `map_to_webnn_op`)
- ONNX graph validation via `validate_onnx_graph` on the shared `ExportIr`/`ExportNode` IR
- `ExportBuilder` fluent config and `ExportOptions` with opset version, quantization, and verification flags
- `export_all` bulk multi-format export with `ExportSummary` reporting
- `QuantExportConfig` (per-channel / bit-width), `verify_export` numeric tolerance checks, `create_zip_archive` pure-Rust ZIP packer
- `.brain` serialization round-trip and CRC tamper detection exercised against `BrainModelFile` from `brain-core`

## Modules

| Module | Description |
|---|---|
| `model` | `ExportModel` parameter container and `ModelExporter` trait |
| `core` | `ExportFormat` (`Onnx | Tflite | CoreMl | WebNn`), `ExportOptions`, `ExportError` |
| `builder` | Fluent `ExportBuilder` (format, opset, build) |
| `config` | `TargetPlatform` and `ExportConfig` |
| `onnx` | `OnnxExporter`, `OnnxConfig`, graph checker, op mapping |
| `tflite` | `TfliteExporter`, `TfliteConfig`, builtin-code mapping |
| `coreml` | `CoreMlExporter`, `CoreMlConfig`, layer-support checks |
| `webnn` | `WebnnExporter`, `WebnnConfig`, op mapping |
| `common` | Shared `ExportIr`/`ExportNode` IR, dtype map, weight helpers |
| `export_all` | `export_all` bulk export + `ExportSummary` |
| `quant_export` | `QuantExportConfig` for quantized export |
| `verify` | `verify_export` numeric tolerance comparison |
| `zip` | Zero-dependency ZIP archive builder |
| `ops` / `ops_supported` | Op metadata and support registry |
| `convert`, `name_gen`, `utils` | Graph conversion, `sanitize_name`, ULEB128/CRC32 helpers |

## Quick Start

```rust
use brain_export::model::{ExportModel, ModelExporter};
use brain_export::onnx::{OnnxConfig, OnnxExporter};
use brain_core::Tensor;

let mut model = ExportModel::new("mlp");
model.add_parameter("fc.weight", Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]));

let exporter = OnnxExporter::new(OnnxConfig::default());
exporter.export(&model, "out/onnx/model.onnx").expect("export");
```

For the `.brain` binary container, `BrainModelFile::new(name)` / `to_bytes()` / `from_bytes()` (with CRC validation) are provided by `brain-core` and covered in `tests/export_test.rs`.

## Testing

```bash
cargo test -p brain-export -j 2
```

## Workspace Role

Depends only on `brain-core` (no external crates). `brain-export` is the interoperability layer of the workspace: it turns the framework's tensor/model representation into consumable ONNX, TFLite, CoreML, and WebNN artifacts for deployment outside the Rust runtime.