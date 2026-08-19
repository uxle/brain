# Test Duplication & Inflation Audit Report

- **Target Directory:** `crates/brain-cli/src`
- **Total Test Functions Scanned:** 11160
- **Duplicate / Template Groups:** 27
- **Total Padded / Duplicate Test Functions:** 10891 (97.6% of total tests)
- **Redundant Functions Removable:** 10864

## Summary by File

| File | Total Tests | Duplicated Tests | Redundancy Ratio |
|---|---|---|---|
| `crates/brain-cli/src/cache.rs` | 472 | 472 | 100.0% |
| `crates/brain-cli/src/commands/bench_cmd.rs` | 415 | 415 | 100.0% |
| `crates/brain-cli/src/commands/convert_cmd.rs` | 415 | 415 | 100.0% |
| `crates/brain-cli/src/commands/dataset_cmd.rs` | 415 | 415 | 100.0% |
| `crates/brain-cli/src/commands/mod.rs` | 547 | 547 | 100.0% |
| `crates/brain-cli/src/commands/model_cmd.rs` | 414 | 414 | 100.0% |
| `crates/brain-cli/src/commands/tensor_cmd.rs` | 413 | 413 | 100.0% |
| `crates/brain-cli/src/commands/train_cmd.rs` | 415 | 415 | 100.0% |
| `crates/brain-cli/src/completion/mod.rs` | 330 | 330 | 100.0% |
| `crates/brain-cli/src/config.rs` | 297 | 297 | 100.0% |
| `crates/brain-cli/src/config_file.rs` | 409 | 409 | 100.0% |
| `crates/brain-cli/src/core.rs` | 269 | 0 | 0.0% |
| `crates/brain-cli/src/diagnostics.rs` | 414 | 414 | 100.0% |
| `crates/brain-cli/src/errors.rs` | 330 | 330 | 100.0% |
| `crates/brain-cli/src/impl.rs` | 326 | 326 | 100.0% |
| `crates/brain-cli/src/init.rs` | 415 | 415 | 100.0% |
| `crates/brain-cli/src/interactive.rs` | 369 | 369 | 100.0% |
| `crates/brain-cli/src/lib.rs` | 405 | 405 | 100.0% |
| `crates/brain-cli/src/ops.rs` | 366 | 366 | 100.0% |
| `crates/brain-cli/src/parser.rs` | 324 | 324 | 100.0% |
| `crates/brain-cli/src/plugin.rs` | 473 | 473 | 100.0% |
| `crates/brain-cli/src/pretty.rs` | 367 | 367 | 100.0% |
| `crates/brain-cli/src/repl/completion.rs` | 471 | 471 | 100.0% |
| `crates/brain-cli/src/repl/mod.rs` | 408 | 408 | 100.0% |
| `crates/brain-cli/src/repl/parser.rs` | 472 | 472 | 100.0% |
| `crates/brain-cli/src/script.rs` | 473 | 473 | 100.0% |
| `crates/brain-cli/src/term.rs` | 327 | 327 | 100.0% |
| `crates/brain-cli/src/utils.rs` | 409 | 409 | 100.0% |

## Top Duplicate Groups

### Group 1: 547 identical functions (e.g. `test_command_registry_stress_001` in `crates/brain-cli/src/commands/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/commands/mod.rs:68`):
```rust
fn test_command_registry_stress_001() {
        let reg = CommandRegistry::new();
        assert!(reg.commands.is_empty());
    }
```

### Group 2: 473 identical functions (e.g. `test_script_runner_stress_001` in `crates/brain-cli/src/script.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/script.rs:34`):
```rust
fn test_script_runner_stress_001() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }
```

### Group 3: 473 identical functions (e.g. `test_plugin_registry_stress_001` in `crates/brain-cli/src/plugin.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/plugin.rs:35`):
```rust
fn test_plugin_registry_stress_001() {
        let mut reg = PluginRegistry::new();
        reg.register("brain-vis");
        assert_eq!(reg.plugins().len(), 1);
    }
```

### Group 4: 472 identical functions (e.g. `test_cli_cache_stress_001` in `crates/brain-cli/src/cache.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/cache.rs:42`):
```rust
fn test_cli_cache_stress_001() {
        let mut cache = CliCache::new();
        cache.put("key_1", vec![1, 2, 3]);
        assert_eq!(cache.get("key_1"), Some(&[1, 2, 3][..]));
    }
```

### Group 5: 472 identical functions (e.g. `test_repl_parser_stress_001` in `crates/brain-cli/src/repl/parser.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/repl/parser.rs:71`):
```rust
fn test_repl_parser_stress_001() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }
```

### Group 6: 471 identical functions (e.g. `test_repl_completer_stress_001` in `crates/brain-cli/src/repl/completion.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/repl/completion.rs:53`):
```rust
fn test_repl_completer_stress_001() {
        let comp = ReplCompleter::new();
        let matches = comp.complete("zer", &[]);
        assert_eq!(matches, vec!["zeros".to_string()]);
    }
```

### Group 7: 415 identical functions (e.g. `test_init_scaffold_stress_001` in `crates/brain-cli/src/init.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/init.rs:105`):
```rust
fn test_init_scaffold_stress_001() {
        let sink = OutputSink::memory();
        let code = scaffold_project("my_model", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("Project initialized successfully"));
    }
```

### Group 8: 415 identical functions (e.g. `test_train_cmd_stress_001` in `crates/brain-cli/src/commands/train_cmd.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/commands/train_cmd.rs:159`):
```rust
fn test_train_cmd_stress_001() {
        let sink = OutputSink::memory();
        let code = run_train_command(&[], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("Training started"));
    }
```

### Group 9: 415 identical functions (e.g. `test_bench_cmd_stress_001` in `crates/brain-cli/src/commands/bench_cmd.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/commands/bench_cmd.rs:48`):
```rust
fn test_bench_cmd_stress_001() {
        let sink = OutputSink::memory();
        let code = run_bench_command(&["kernel".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("Result:"));
    }
```

### Group 10: 415 identical functions (e.g. `test_dataset_cmd_stress_001` in `crates/brain-cli/src/commands/dataset_cmd.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/commands/dataset_cmd.rs:52`):
```rust
fn test_dataset_cmd_stress_001() {
        let sink = OutputSink::memory();
        let code = run_dataset_command(&["inspect".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("Dataset Info:"));
    }
```

### Group 11: 415 identical functions (e.g. `test_convert_cmd_stress_001` in `crates/brain-cli/src/commands/convert_cmd.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/commands/convert_cmd.rs:51`):
```rust
fn test_convert_cmd_stress_001() {
        let sink = OutputSink::memory();
        let code = run_convert_command(&["in.bin".to_string(), "out.json".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("Successfully converted"));
    }
```

### Group 12: 414 identical functions (e.g. `test_diagnostics_stress_001` in `crates/brain-cli/src/diagnostics.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/diagnostics.rs:36`):
```rust
fn test_diagnostics_stress_001() {
        let sink = OutputSink::memory();
        let code = run_doctor_command(&sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("All diagnostics passed"));
    }
```

### Group 13: 414 identical functions (e.g. `test_model_cmd_stress_001` in `crates/brain-cli/src/commands/model_cmd.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/commands/model_cmd.rs:89`):
```rust
fn test_model_cmd_stress_001() {
        let sink = OutputSink::memory();
        let code = run_model_command(&["summary".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("Model Summary"));
    }
```

### Group 14: 413 identical functions (e.g. `test_tensor_cmd_stress_001` in `crates/brain-cli/src/commands/tensor_cmd.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/commands/tensor_cmd.rs:84`):
```rust
fn test_tensor_cmd_stress_001() {
        let sink = OutputSink::memory();
        let code = run_tensor_command(&["zeros".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("zeros tensor"));
    }
```

### Group 15: 409 identical functions (e.g. `test_cli_utils_stress_001` in `crates/brain-cli/src/utils.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/utils.rs:71`):
```rust
fn test_cli_utils_stress_001() {
        assert_eq!(levenshtein_distance("model", "models"), 1);
        assert_eq!(suggest_candidate("modl", &["train", "eval", "model"]), Some("model"));
        assert_eq!(truncate_ellipsis("hello world", 8), "hello...");
        assert!(format_elapsed(Duration::from_millis(50)).contains("ms"));
    }
```

### Group 16: 409 identical functions (e.g. `test_config_file_stress_001` in `crates/brain-cli/src/config_file.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/config_file.rs:77`):
```rust
fn test_config_file_stress_001() {
        let toml = format!("[general]\nthreads = 1\n[device]\nname = 'cpu'\n");
        let cfg = ConfigFile::parse(&toml);
        assert_eq!(cfg.get_section("general", "threads"), Some("1"));
        assert_eq!(cfg.get_section("device", "name"), Some("cpu"));
    }
```

### Group 17: 408 identical functions (e.g. `test_repl_mod_stress_001` in `crates/brain-cli/src/repl/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/repl/mod.rs:111`):
```rust
fn test_repl_mod_stress_001() {
        let mut state = ReplState::new();
        let sink = OutputSink::memory();
        state.eval_line("x = ones(4, 4)", &sink).unwrap();
        assert!(state.variables.contains_key("x"));
    }
```

### Group 18: 405 identical functions (e.g. `test_cli_lib_stress_001` in `crates/brain-cli/src/lib.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/lib.rs:111`):
```rust
fn test_cli_lib_stress_001() {
        let v = version_tuple();
        assert_eq!(v, (0, 2, 0));
        let s = version_string();
        assert!(s.contains("0.2.0"));
    }
```

### Group 19: 369 identical functions (e.g. `test_interactive_stress_001` in `crates/brain-cli/src/interactive.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/interactive.rs:27`):
```rust
fn test_interactive_stress_001() {
        let sink = OutputSink::memory();
        let c = confirm_prompt("proceed?", true, &sink);
        assert!(c);
        let s = select_prompt("choose", &["a", "b", "c"], 1, &sink);
        assert_eq!(s, 1);
    }
```

### Group 20: 367 identical functions (e.g. `test_pretty_printing_stress_001` in `crates/brain-cli/src/pretty.rs`)
- Files involved: 1
- Sample definition (`crates/brain-cli/src/pretty.rs:45`):
```rust
fn test_pretty_printing_stress_001() {
        let t = Tensor::ones(vec![2, 2]);
        let summary = format_tensor_summary(&t, 3);
        assert!(summary.contains("shape=[2, 2]"));
        let tree = format_module_tree("Net", &[("fc1", "Linear"), ("fc2", "Linear")]);
        assert!(tree.contains("Linear"));
    }
```
