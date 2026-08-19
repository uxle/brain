# Test Duplication & Inflation Audit Report

- **Target Directory:** `crates/brain-export/src`
- **Total Test Functions Scanned:** 13509
- **Duplicate / Template Groups:** 28
- **Total Padded / Duplicate Test Functions:** 13509 (100.0% of total tests)
- **Redundant Functions Removable:** 13481

## Summary by File

| File | Total Tests | Duplicated Tests | Redundancy Ratio |
|---|---|---|---|
| `crates/brain-export/src/builder.rs` | 330 | 330 | 100.0% |
| `crates/brain-export/src/common/dtype_map.rs` | 551 | 551 | 100.0% |
| `crates/brain-export/src/common/ir.rs` | 300 | 300 | 100.0% |
| `crates/brain-export/src/common/mod.rs` | 555 | 555 | 100.0% |
| `crates/brain-export/src/common/weights.rs` | 475 | 475 | 100.0% |
| `crates/brain-export/src/config.rs` | 472 | 472 | 100.0% |
| `crates/brain-export/src/convert.rs` | 553 | 553 | 100.0% |
| `crates/brain-export/src/core.rs` | 471 | 471 | 100.0% |
| `crates/brain-export/src/coreml/mod.rs` | 471 | 471 | 100.0% |
| `crates/brain-export/src/coreml/ops.rs` | 555 | 555 | 100.0% |
| `crates/brain-export/src/export_all.rs` | 414 | 414 | 100.0% |
| `crates/brain-export/src/impl.rs` | 475 | 475 | 100.0% |
| `crates/brain-export/src/lib.rs` | 404 | 404 | 100.0% |
| `crates/brain-export/src/model/mod.rs` | 472 | 472 | 100.0% |
| `crates/brain-export/src/name_gen.rs` | 553 | 553 | 100.0% |
| `crates/brain-export/src/onnx/checker.rs` | 554 | 554 | 100.0% |
| `crates/brain-export/src/onnx/mod.rs` | 470 | 470 | 100.0% |
| `crates/brain-export/src/onnx/ops.rs` | 553 | 553 | 100.0% |
| `crates/brain-export/src/ops.rs` | 554 | 554 | 100.0% |
| `crates/brain-export/src/ops_supported.rs` | 553 | 553 | 100.0% |
| `crates/brain-export/src/quant_export.rs` | 474 | 474 | 100.0% |
| `crates/brain-export/src/tflite/mod.rs` | 471 | 471 | 100.0% |
| `crates/brain-export/src/tflite/ops.rs` | 554 | 554 | 100.0% |
| `crates/brain-export/src/utils.rs` | 366 | 366 | 100.0% |
| `crates/brain-export/src/verify.rs` | 474 | 474 | 100.0% |
| `crates/brain-export/src/webnn/mod.rs` | 473 | 473 | 100.0% |
| `crates/brain-export/src/webnn/ops.rs` | 553 | 553 | 100.0% |
| `crates/brain-export/src/zip.rs` | 409 | 409 | 100.0% |

## Top Duplicate Groups

### Group 1: 555 identical functions (e.g. `test_common_mod_stress_001` in `crates/brain-export/src/common/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/common/mod.rs:20`):
```rust
fn test_common_mod_stress_001() {
        let ir = ExportIr::new("graph");
        assert_eq!(ir.name, "graph");
    }
```

### Group 2: 555 identical functions (e.g. `test_coreml_ops_stress_001` in `crates/brain-export/src/coreml/ops.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/coreml/ops.rs:20`):
```rust
fn test_coreml_ops_stress_001() {
        assert!(is_coreml_layer_supported("Convolution"));
        assert!(is_coreml_layer_supported("InnerProduct"));
    }
```

### Group 3: 554 identical functions (e.g. `test_export_ops_stress_001` in `crates/brain-export/src/ops.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/ops.rs:22`):
```rust
fn test_export_ops_stress_001() {
        assert!(is_op_supported("Add", ExportFormat::Onnx));
        assert!(is_op_supported("Conv2d", ExportFormat::Tflite));
    }
```

### Group 4: 554 identical functions (e.g. `test_checker_stress_001` in `crates/brain-export/src/onnx/checker.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/onnx/checker.rs:23`):
```rust
fn test_checker_stress_001() {
        let ir = ExportIr::new("valid_graph");
        assert!(validate_onnx_graph(&ir).is_ok());
    }
```

### Group 5: 554 identical functions (e.g. `test_tflite_ops_stress_001` in `crates/brain-export/src/tflite/ops.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/tflite/ops.rs:26`):
```rust
fn test_tflite_ops_stress_001() {
        assert_eq!(map_to_tflite_builtin_code("Add"), Some(0));
        assert_eq!(map_to_tflite_builtin_code("Relu"), Some(19));
    }
```

### Group 6: 553 identical functions (e.g. `test_convert_stress_001` in `crates/brain-export/src/convert.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/convert.rs:27`):
```rust
fn test_convert_stress_001() {
        let rep = ConversionReport::new(1);
        assert_eq!(rep.num_nodes_converted, 1);
    }
```

### Group 7: 553 identical functions (e.g. `test_ops_supported_stress_001` in `crates/brain-export/src/ops_supported.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/ops_supported.rs:27`):
```rust
fn test_ops_supported_stress_001() {
        let rep = SupportedOpsReport::new(50);
        assert_eq!(rep.supported_count, 50);
    }
```

### Group 8: 553 identical functions (e.g. `test_name_gen_stress_001` in `crates/brain-export/src/name_gen.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/name_gen.rs:28`):
```rust
fn test_name_gen_stress_001() {
        let s = sanitize_name("layer.1/weight");
        assert_eq!(s, "layer_1_weight");
    }
```

### Group 9: 553 identical functions (e.g. `test_onnx_ops_stress_001` in `crates/brain-export/src/onnx/ops.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/onnx/ops.rs:27`):
```rust
fn test_onnx_ops_stress_001() {
        assert_eq!(map_to_onnx_op("Add"), Some("Add"));
        assert_eq!(map_to_onnx_op("Conv2d"), Some("Conv"));
    }
```

### Group 10: 553 identical functions (e.g. `test_webnn_ops_stress_001` in `crates/brain-export/src/webnn/ops.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/webnn/ops.rs:27`):
```rust
fn test_webnn_ops_stress_001() {
        assert_eq!(map_to_webnn_op("Add"), Some("add"));
        assert_eq!(map_to_webnn_op("Conv2d"), Some("conv2d"));
    }
```

### Group 11: 551 identical functions (e.g. `test_dtype_map_stress_001` in `crates/brain-export/src/common/dtype_map.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/common/dtype_map.rs:42`):
```rust
fn test_dtype_map_stress_001() {
        assert_eq!(map_dtype_to_onnx(DTypeKind::Float32), 1);
        assert_eq!(map_dtype_to_tflite(DTypeKind::Float32), 0);
    }
```

### Group 12: 475 identical functions (e.g. `test_export_impl_stress_001` in `crates/brain-export/src/impl.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/impl.rs:25`):
```rust
fn test_export_impl_stress_001() {
        let model = ExportModel::new("test_model");
        let res = export_model(&model, ExportFormat::Onnx, "model.onnx", &ExportOptions::default());
        assert!(res.is_ok());
    }
```

### Group 13: 475 identical functions (e.g. `test_weights_stress_001` in `crates/brain-export/src/common/weights.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/common/weights.rs:24`):
```rust
fn test_weights_stress_001() {
        let t = Tensor::zeros(vec![2, 2]);
        let bytes = serialize_weights_f32(&t);
        assert_eq!(bytes.len(), 16);
    }
```

### Group 14: 474 identical functions (e.g. `test_verify_stress_001` in `crates/brain-export/src/verify.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/verify.rs:28`):
```rust
fn test_verify_stress_001() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        assert!(verify_export(&t1, &t2, 1e-4).is_ok());
    }
```

### Group 15: 474 identical functions (e.g. `test_quant_export_stress_001` in `crates/brain-export/src/quant_export.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/quant_export.rs:29`):
```rust
fn test_quant_export_stress_001() {
        let cfg = QuantExportConfig::new(true, 8);
        assert!(cfg.per_channel);
        assert_eq!(cfg.bit_width, 8);
    }
```

### Group 16: 473 identical functions (e.g. `test_webnn_mod_stress_001` in `crates/brain-export/src/webnn/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/webnn/mod.rs:39`):
```rust
fn test_webnn_mod_stress_001() {
        let exp = WebnnExporter::new(WebnnConfig::default());
        let m = ExportModel::new("m");
        assert!(exp.export(&m, "m.webnn.json").is_ok());
    }
```

### Group 17: 472 identical functions (e.g. `test_export_config_stress_001` in `crates/brain-export/src/config.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/config.rs:42`):
```rust
fn test_export_config_stress_001() {
        let cfg = ExportConfig::default();
        assert_eq!(cfg.format, ExportFormat::Onnx);
        assert!(cfg.optimize_graph);
    }
```

### Group 18: 472 identical functions (e.g. `test_export_model_stress_001` in `crates/brain-export/src/model/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/model/mod.rs:42`):
```rust
fn test_export_model_stress_001() {
        let mut m = ExportModel::new("test_model");
        m.add_parameter("w1", Tensor::zeros(vec![4, 4]));
        assert_eq!(m.parameters.len(), 1);
    }
```

### Group 19: 471 identical functions (e.g. `test_export_core_stress_001` in `crates/brain-export/src/core.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/core.rs:50`):
```rust
fn test_export_core_stress_001() {
        let opt = ExportOptions::default();
        assert_eq!(opt.format, ExportFormat::Onnx);
        assert_eq!(opt.opset_version, 17);
    }
```

### Group 20: 471 identical functions (e.g. `test_tflite_mod_stress_001` in `crates/brain-export/src/tflite/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-export/src/tflite/mod.rs:49`):
```rust
fn test_tflite_mod_stress_001() {
        let exp = TfliteExporter::new(TfliteConfig::default());
        let m = ExportModel::new("m");
        assert!(exp.export(&m, "m.tflite").is_ok());
    }
```
