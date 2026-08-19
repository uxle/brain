# `brain-cli`

Developer-grade command-line suite for the Brain framework: subcommands, REPL, scripting, and diagnostics.

## Overview

`brain-cli` is the library behind the `brain` binary. It dispatches 15+ subcommands — project scaffolding (`new`, `make`, `init`), tensor/model/benchmark inspection (`tensor`, `model`, `bench`, `space`, `check`), training (`train`, `run`), data/format conversion (`dataset`, `convert`), an interactive math REPL with a `chat`/`chatbot` brain-mind mode, script files, and a `doctor` diagnostics command. Output is routed through an injectable `OutputSink` so everything is unit-testable.

## Features

- **Subcommands** — `tensor`, `bench`, `model`, `train`, `make`, `run`, `space`, `chat`, `new`, `check`, `script`, `dataset`, `convert`, `doctor`, `repl`, `init`, plus `--version`/`--help`.
- **REPL** — interactive expression parser with completion and shell-completion generation (`completion` module).
- **Scripting** — run Brain script files via `script::run_script`.
- **Config & diagnostics** — `CliConfig`, `Verbosity`, `ColorChoice`, `ConfigFile` persistence, and system `doctor` inspection.
- **Testable I/O** — `run_cli(args, &OutputSink)` returns an `ExitCode`; `OutputSink::stdout()` / `OutputSink::memory()`.
- **Model & training glue** — wraps `brain-train` trainers (`train_cmd`), `brain-optim` autopilot, and dataset generation.

## Modules

| Module | Description |
|---|---|
| `impl` | `run_cli` dispatcher over subcommands |
| `core` | `ExitCode`, `OutputSink`, `OutputFormat`, `CommandSpec` |
| `commands` | `make`, `run`, `train`, `check`, `tensor`, `model`, `bench`, `space`, `dataset`, `convert` commands |
| `repl` | Interactive REPL, parser, completion |
| `script` | Script-file execution |
| `config` / `config_file` | CLI configuration and persistence |
| `parser` | Expression parser used by REPL/scripts |
| `diagnostics` / `doctor` | System inspection |
| `completion` | Shell completion generation |
| `pretty` / `term` / `utils` | Output formatting and terminal helpers |

## Quick Start

```rust
use brain_cli::core::OutputSink;
use brain_cli::run_cli;

let sink = OutputSink::memory();
let exit = run_cli(&["--version".to_string()], &sink);
assert!(exit.is_success());
assert!(sink.captured().unwrap().contains("brain-cli v"));
```

From a shell, the binary is used as `brain make`, `brain run`, `brain train`, `brain check`, `brain doctor`, `brain repl`, …

## Testing

```bash
cargo test -p brain-cli -j 2
```

Tests cover command dispatch, script/REPL equivalence, `space` output, and `brain_mind` conversations.

## Workspace Role

Depends on `brain-core`, `brain-autograd`, `brain-loss`, `brain-optim`, `brain-train`, and `brain-transformer`. Consumer: the `brain` binary (via its `cli` feature).