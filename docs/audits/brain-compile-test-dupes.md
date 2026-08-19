# Test Duplication & Inflation Audit Report

- **Target Directory:** `crates/brain-compile/src`
- **Total Test Functions Scanned:** 14073
- **Duplicate / Template Groups:** 33
- **Total Padded / Duplicate Test Functions:** 14073 (100.0% of total tests)
- **Redundant Functions Removable:** 14040

## Summary by File

| File | Total Tests | Duplicated Tests | Redundancy Ratio |
|---|---|---|---|
| `crates/brain-compile/src/analyze.rs` | 254 | 254 | 100.0% |
| `crates/brain-compile/src/backend/cuda.rs` | 475 | 475 | 100.0% |
| `crates/brain-compile/src/backend/interp.rs` | 410 | 410 | 100.0% |
| `crates/brain-compile/src/backend/llvm.rs` | 475 | 475 | 100.0% |
| `crates/brain-compile/src/backend/mod.rs` | 416 | 416 | 100.0% |
| `crates/brain-compile/src/backend/scalar.rs` | 475 | 475 | 100.0% |
| `crates/brain-compile/src/backend/tensor.rs` | 414 | 414 | 100.0% |
| `crates/brain-compile/src/builder.rs` | 273 | 273 | 100.0% |
| `crates/brain-compile/src/compute.rs` | 474 | 474 | 100.0% |
| `crates/brain-compile/src/config.rs` | 549 | 549 | 100.0% |
| `crates/brain-compile/src/core.rs` | 295 | 295 | 100.0% |
| `crates/brain-compile/src/exec.rs` | 414 | 414 | 100.0% |
| `crates/brain-compile/src/export_ir.rs` | 368 | 368 | 100.0% |
| `crates/brain-compile/src/helper.rs` | 552 | 552 | 100.0% |
| `crates/brain-compile/src/impl.rs` | 414 | 414 | 100.0% |
| `crates/brain-compile/src/ir/mod.rs` | 294 | 294 | 100.0% |
| `crates/brain-compile/src/ir/ops.rs` | 469 | 469 | 100.0% |
| `crates/brain-compile/src/ir/verify.rs` | 547 | 547 | 100.0% |
| `crates/brain-compile/src/jit.rs` | 367 | 367 | 100.0% |
| `crates/brain-compile/src/lib.rs` | 404 | 404 | 100.0% |
| `crates/brain-compile/src/ops.rs` | 470 | 470 | 100.0% |
| `crates/brain-compile/src/passes/broadcast.rs` | 415 | 415 | 100.0% |
| `crates/brain-compile/src/passes/dce.rs` | 415 | 415 | 100.0% |
| `crates/brain-compile/src/passes/fold.rs` | 414 | 414 | 100.0% |
| `crates/brain-compile/src/passes/fusion.rs` | 415 | 415 | 100.0% |
| `crates/brain-compile/src/passes/layout.rs` | 415 | 415 | 100.0% |
| `crates/brain-compile/src/passes/mod.rs` | 409 | 409 | 100.0% |
| `crates/brain-compile/src/plan.rs` | 473 | 473 | 100.0% |
| `crates/brain-compile/src/process.rs` | 475 | 475 | 100.0% |
| `crates/brain-compile/src/profiler.rs` | 474 | 474 | 100.0% |
| `crates/brain-compile/src/schedule.rs` | 474 | 474 | 100.0% |
| `crates/brain-compile/src/transform.rs` | 555 | 555 | 100.0% |
| `crates/brain-compile/src/utils.rs` | 330 | 330 | 100.0% |

## Top Duplicate Groups

### Group 1: 555 identical functions (e.g. `test_transform_stress_001` in `crates/brain-compile/src/transform.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/transform.rs:20`):
```rust
fn test_transform_stress_001() {
        let mut g = IrGraph::new();
        assert!(!apply_algebraic_rewrites(&mut g));
    }
```

### Group 2: 552 identical functions (e.g. `test_helper_broadcast_stress_001` in `crates/brain-compile/src/helper.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/helper.rs:36`):
```rust
fn test_helper_broadcast_stress_001() {
        let shape = compute_broadcast_shape(&[2, 1], &[1, 4]).unwrap();
        assert_eq!(shape, vec![2, 4]);
    }
```

### Group 3: 549 identical functions (e.g. `test_compiler_config_stress_001` in `crates/brain-compile/src/config.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/config.rs:51`):
```rust
fn test_compiler_config_stress_001() {
        let cfg = CompilerConfig::new().with_cache_capacity(1 + 10);
        assert_eq!(cfg.cache.max_entries, 1 + 10);
    }
```

### Group 4: 547 identical functions (e.g. `test_ir_verify_stress_001` in `crates/brain-compile/src/ir/verify.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/ir/verify.rs:67`):
```rust
fn test_ir_verify_stress_001() {
        let g = IrGraph::new();
        assert!(verify_graph(&g).is_ok());
    }
```

### Group 5: 475 identical functions (e.g. `test_process_pipeline_stress_001` in `crates/brain-compile/src/process.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/process.rs:21`):
```rust
fn test_process_pipeline_stress_001() {
        let mut g = IrGraph::new();
        let res = run_pipeline_stage("test", &mut g);
        assert!(res.is_ok());
    }
```

### Group 6: 475 identical functions (e.g. `test_scalar_codegen_stress_001` in `crates/brain-compile/src/backend/scalar.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/backend/scalar.rs:25`):
```rust
fn test_scalar_codegen_stress_001() {
        let g = IrGraph::new();
        let code = generate_rust_kernel(&g, "my_kernel");
        assert!(code.contains("pub fn my_kernel"));
    }
```

### Group 7: 475 identical functions (e.g. `test_cuda_codegen_stress_001` in `crates/brain-compile/src/backend/cuda.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/backend/cuda.rs:25`):
```rust
fn test_cuda_codegen_stress_001() {
        let g = IrGraph::new();
        let cu = generate_cuda_kernel(&g, "cuda_kernel");
        assert!(cu.contains("__global__ void cuda_kernel"));
    }
```

### Group 8: 475 identical functions (e.g. `test_llvm_codegen_stress_001` in `crates/brain-compile/src/backend/llvm.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/backend/llvm.rs:24`):
```rust
fn test_llvm_codegen_stress_001() {
        let g = IrGraph::new();
        let ll = generate_llvm_ir(&g, "test_mod");
        assert!(ll.contains("ModuleID = 'test_mod'"));
    }
```

### Group 9: 474 identical functions (e.g. `test_compute_analysis_stress_001` in `crates/brain-compile/src/compute.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/compute.rs:27`):
```rust
fn test_compute_analysis_stress_001() {
        let g = IrGraph::new();
        let lt = analyze_tensor_lifetimes(&g);
        assert!(lt.is_empty());
    }
```

### Group 10: 474 identical functions (e.g. `test_profiler_stress_001` in `crates/brain-compile/src/profiler.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/profiler.rs:31`):
```rust
fn test_profiler_stress_001() {
        let g = IrGraph::new();
        let rep = ProfileReport::profile(&g);
        assert_eq!(rep.op_counts, 0);
    }
```

### Group 11: 474 identical functions (e.g. `test_schedule_stress_001` in `crates/brain-compile/src/schedule.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/schedule.rs:30`):
```rust
fn test_schedule_stress_001() {
        let g = IrGraph::new();
        let sched = SchedulePlan::compute_schedule(&g);
        assert!(sched.execution_order.is_empty());
    }
```

### Group 12: 473 identical functions (e.g. `test_memory_plan_stress_001` in `crates/brain-compile/src/plan.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/plan.rs:37`):
```rust
fn test_memory_plan_stress_001() {
        let g = IrGraph::new();
        let plan = MemoryPlan::create_plan(&g);
        assert_eq!(plan.peak_memory_bytes, 0);
    }
```

### Group 13: 470 identical functions (e.g. `test_compile_ops_stress_001` in `crates/brain-compile/src/ops.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/ops.rs:55`):
```rust
fn test_compile_ops_stress_001() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }
```

### Group 14: 469 identical functions (e.g. `test_ir_ops_catalog_stress_001` in `crates/brain-compile/src/ir/ops.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/ir/ops.rs:63`):
```rust
fn test_ir_ops_catalog_stress_001() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }
```

### Group 15: 416 identical functions (e.g. `test_backend_mod_stress_001` in `crates/brain-compile/src/backend/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/backend/mod.rs:21`):
```rust
fn test_backend_mod_stress_001() {
        let interp = Interpreter::new();
        let tensor_b = TensorBackend::new();
        assert_eq!(interp.name(), "interpreter");
        assert_eq!(tensor_b.name(), "tensor");
    }
```

### Group 16: 415 identical functions (e.g. `test_dce_stress_001` in `crates/brain-compile/src/passes/dce.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/passes/dce.rs:29`):
```rust
fn test_dce_stress_001() {
        let pass = DeadCodeEliminationPass;
        let mut g = IrGraph::new();
        let res = pass.run(&mut g);
        assert!(res.is_ok());
    }
```

### Group 17: 415 identical functions (e.g. `test_kernel_fusion_stress_001` in `crates/brain-compile/src/passes/fusion.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/passes/fusion.rs:29`):
```rust
fn test_kernel_fusion_stress_001() {
        let pass = KernelFusionPass;
        let mut g = IrGraph::new();
        let res = pass.run(&mut g);
        assert!(res.is_ok());
    }
```

### Group 18: 415 identical functions (e.g. `test_broadcast_pass_stress_001` in `crates/brain-compile/src/passes/broadcast.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/passes/broadcast.rs:29`):
```rust
fn test_broadcast_pass_stress_001() {
        let pass = BroadcastEliminationPass;
        let mut g = IrGraph::new();
        let res = pass.run(&mut g);
        assert!(res.is_ok());
    }
```

### Group 19: 415 identical functions (e.g. `test_layout_pass_stress_001` in `crates/brain-compile/src/passes/layout.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/passes/layout.rs:29`):
```rust
fn test_layout_pass_stress_001() {
        let pass = LayoutOptimizationPass;
        let mut g = IrGraph::new();
        let res = pass.run(&mut g);
        assert!(res.is_ok());
    }
```

### Group 20: 414 identical functions (e.g. `test_compile_impl_stress_001` in `crates/brain-compile/src/impl.rs`)
- Files involved: 1
- Sample definition (`crates/brain-compile/src/impl.rs:32`):
```rust
fn test_compile_impl_stress_001() {
        let g = IrGraph::new();
        let opts = CompileOptions::new();
        let res = compile_graph(&g, &opts);
        assert!(res.is_ok());
    }
```
