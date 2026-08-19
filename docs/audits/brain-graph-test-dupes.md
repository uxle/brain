# Test Duplication & Inflation Audit Report

- **Target Directory:** `crates/brain-graph/src`
- **Total Test Functions Scanned:** 7688
- **Duplicate / Template Groups:** 31
- **Total Padded / Duplicate Test Functions:** 7688 (100.0% of total tests)
- **Redundant Functions Removable:** 7657

## Summary by File

| File | Total Tests | Duplicated Tests | Redundancy Ratio |
|---|---|---|---|
| `crates/brain-graph/src/analyze.rs` | 236 | 236 | 100.0% |
| `crates/brain-graph/src/builder.rs` | 218 | 218 | 100.0% |
| `crates/brain-graph/src/clone.rs` | 235 | 235 | 100.0% |
| `crates/brain-graph/src/compute.rs` | 235 | 235 | 100.0% |
| `crates/brain-graph/src/config.rs` | 294 | 294 | 100.0% |
| `crates/brain-graph/src/core.rs` | 232 | 232 | 100.0% |
| `crates/brain-graph/src/diff.rs` | 413 | 413 | 100.0% |
| `crates/brain-graph/src/dot.rs` | 235 | 235 | 100.0% |
| `crates/brain-graph/src/helper.rs` | 253 | 253 | 100.0% |
| `crates/brain-graph/src/impl_.rs` | 415 | 415 | 100.0% |
| `crates/brain-graph/src/interp.rs` | 163 | 163 | 100.0% |
| `crates/brain-graph/src/ir/mod.rs` | 292 | 292 | 100.0% |
| `crates/brain-graph/src/ir/ops.rs` | 323 | 323 | 100.0% |
| `crates/brain-graph/src/ir/shape_infer.rs` | 218 | 218 | 100.0% |
| `crates/brain-graph/src/ir/verify.rs` | 297 | 297 | 100.0% |
| `crates/brain-graph/src/json.rs` | 236 | 236 | 100.0% |
| `crates/brain-graph/src/ops.rs` | 234 | 234 | 100.0% |
| `crates/brain-graph/src/optimize.rs` | 253 | 253 | 100.0% |
| `crates/brain-graph/src/passes/const_fold.rs` | 205 | 205 | 100.0% |
| `crates/brain-graph/src/passes/cse.rs` | 193 | 193 | 100.0% |
| `crates/brain-graph/src/passes/dead_code.rs` | 182 | 182 | 100.0% |
| `crates/brain-graph/src/passes/fusion.rs` | 182 | 182 | 100.0% |
| `crates/brain-graph/src/passes/inplace.rs` | 235 | 235 | 100.0% |
| `crates/brain-graph/src/passes/layout.rs` | 206 | 206 | 100.0% |
| `crates/brain-graph/src/passes/mod.rs` | 409 | 409 | 100.0% |
| `crates/brain-graph/src/process.rs` | 255 | 255 | 100.0% |
| `crates/brain-graph/src/profile.rs` | 235 | 235 | 100.0% |
| `crates/brain-graph/src/schedule.rs` | 183 | 183 | 100.0% |
| `crates/brain-graph/src/topology.rs` | 180 | 180 | 100.0% |
| `crates/brain-graph/src/transform.rs` | 206 | 206 | 100.0% |
| `crates/brain-graph/src/utils.rs` | 235 | 235 | 100.0% |

## Top Duplicate Groups

### Group 1: 415 identical functions (e.g. `test_impl_stress_001` in `crates/brain-graph/src/impl_.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/impl_.rs:29`):
```rust
fn test_impl_stress_001() {
        let mut g = GraphIr::new(&format!("impl_g_1"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }
```

### Group 2: 413 identical functions (e.g. `test_diff_stress_001` in `crates/brain-graph/src/diff.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/diff.rs:44`):
```rust
fn test_diff_stress_001() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }
```

### Group 3: 409 identical functions (e.g. `test_passes_mod_stress_001` in `crates/brain-graph/src/passes/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/passes/mod.rs:71`):
```rust
fn test_passes_mod_stress_001() {
        let mut pm = PassManager::new();
        let mut g = GraphIr::new(&format!("pass_1"));
        let iters = pm.run(&mut g, 5);
        assert!(iters.is_ok());
    }
```

### Group 4: 323 identical functions (e.g. `test_ops_stress_001` in `crates/brain-graph/src/ir/ops.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/ir/ops.rs:119`):
```rust
fn test_ops_stress_001() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }
```

### Group 5: 297 identical functions (e.g. `test_verify_stress_001` in `crates/brain-graph/src/ir/verify.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/ir/verify.rs:77`):
```rust
fn test_verify_stress_001() {
        let mut g = GraphIr::new(&format!("verify_1"));
        let v_in = g.add_value("in", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.inputs.push(v_in);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v_in], vec![v_out]);
        g.outputs.push(v_out);
        assert!(verify_graph(&g).is_ok());
    }
```

### Group 6: 294 identical functions (e.g. `test_config_stress_001` in `crates/brain-graph/src/config.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/config.rs:113`):
```rust
fn test_config_stress_001() {
        let cfg = GraphConfig::for_opt_level(OptLevel::O2);
        assert!(cfg.validate().is_ok());
        assert!(cfg.enable_fusion);
        let s = cfg.summary();
        assert!(!s.is_empty());
        let o0 = GraphConfig::for_opt_level(OptLevel::O0);
        assert!(!o0.enable_cse);
    }
```

### Group 7: 292 identical functions (e.g. `test_ir_mod_stress_001` in `crates/brain-graph/src/ir/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/ir/mod.rs:130`):
```rust
fn test_ir_mod_stress_001() {
        let mut ir = GraphIr::new(&format!("ir_1"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }
```

### Group 8: 255 identical functions (e.g. `test_process_stress_001` in `crates/brain-graph/src/process.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/process.rs:27`):
```rust
fn test_process_stress_001() {
        let mut g = GraphIr::new("proc_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let res = process_with_verification(&mut g, |_g| Ok(()));
        assert!(res.is_ok());
    }
```

### Group 9: 253 identical functions (e.g. `test_helper_stress_001` in `crates/brain-graph/src/helper.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/helper.rs:60`):
```rust
fn test_helper_stress_001() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }
```

### Group 10: 253 identical functions (e.g. `test_optimize_stress_001` in `crates/brain-graph/src/optimize.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/optimize.rs:59`):
```rust
fn test_optimize_stress_001() {
        let mut g = GraphIr::new("opt_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let report = optimize(&mut g, OptLevel::O2).unwrap();
        assert_eq!(report.final_nodes, 1);
    }
```

### Group 11: 236 identical functions (e.g. `test_analyze_stress_001` in `crates/brain-graph/src/analyze.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/analyze.rs:35`):
```rust
fn test_analyze_stress_001() {
        let mut g = GraphIr::new("analyze_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        assert!(!analyze_cycles(&g));
        let par = analyze_parallelism(&g);
        assert!(par > 0.0);
    }
```

### Group 12: 236 identical functions (e.g. `test_json_stress_001` in `crates/brain-graph/src/json.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/json.rs:41`):
```rust
fn test_json_stress_001() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }
```

### Group 13: 235 identical functions (e.g. `test_compute_stress_001` in `crates/brain-graph/src/compute.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/compute.rs:54`):
```rust
fn test_compute_stress_001() {
        let mut g = GraphIr::new("comp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let costs = compute_costs(&g);
        assert!(costs.total_flops > 0);
        assert!(costs.total_memory_traffic_bytes > 0);
    }
```

### Group 14: 235 identical functions (e.g. `test_utils_stress_001` in `crates/brain-graph/src/utils.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/utils.rs:61`):
```rust
fn test_utils_stress_001() {
        let mut gen = IdGenerator::new();
        for i in 0..6 {
            assert_eq!(gen.next(), i);
        }
        let san = sanitize_name("conv2d.weight:0");
        assert_eq!(san, "conv2d_weight_0");
        let h = hash_attributes(&[("op", "matmul"), ("axis", "1")]);
        assert_ne!(h, 0);
        let summ = format_graph_summary(10, 15, "test");
        assert!(summ.contains("10 nodes"));
    }
```

### Group 15: 235 identical functions (e.g. `test_dot_stress_001` in `crates/brain-graph/src/dot.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/dot.rs:54`):
```rust
fn test_dot_stress_001() {
        let mut g = GraphIr::new("dot_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let dot = to_dot(&g);
        assert!(dot.contains("digraph \"dot_test\""));
        assert!(dot.contains("node_0"));
    }
```

### Group 16: 235 identical functions (e.g. `test_clone_stress_001` in `crates/brain-graph/src/clone.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/clone.rs:50`):
```rust
fn test_clone_stress_001() {
        let mut g = GraphIr::new("orig");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("n0", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let sub = clone_subgraph(&g, &[0]);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.values.len(), 2);
    }
```

### Group 17: 235 identical functions (e.g. `test_profile_stress_001` in `crates/brain-graph/src/profile.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/profile.rs:60`):
```rust
fn test_profile_stress_001() {
        let mut g = GraphIr::new("prof_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2, 4]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let prof = profile_graph(&g);
        assert_eq!(prof.total_nodes, 1);
        assert!(prof.peak_memory_bytes > 0);
    }
```

### Group 18: 235 identical functions (e.g. `test_inplace_stress_001` in `crates/brain-graph/src/passes/inplace.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/passes/inplace.rs:59`):
```rust
fn test_inplace_stress_001() {
        let mut g = GraphIr::new("inplace_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("act", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let plan = plan_inplace_operations(&g).unwrap();
        assert_eq!(plan.in_place_pairs.len(), 1);
    }
```

### Group 19: 234 identical functions (e.g. `test_ops_stress_001` in `crates/brain-graph/src/ops.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/ops.rs:93`):
```rust
fn test_ops_stress_001() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }
```

### Group 20: 232 identical functions (e.g. `test_core_stress_001` in `crates/brain-graph/src/core.rs`)
- Files involved: 1
- Sample definition (`crates/brain-graph/src/core.rs:99`):
```rust
fn test_core_stress_001() {
        let s = Shape::new(vec![2, 2]);
        assert_eq!(s.rank(), 2);
        assert_eq!(s.num_elements(), (2) * (2));
        let meta = GraphMetadata {
            name: format!("graph_1"),
            version: 1,
            author: "Brain Team".into(),
            target_device: DeviceKind::Cpu,
        };
        assert_eq!(meta.version, 1);
    }
```
