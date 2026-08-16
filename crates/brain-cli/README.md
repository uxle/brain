# `brain-cli` (v0.2.0)

> Interactive CLI, REPL Environment, Shell Completions, Model Inspection, and Command Runner.

## Overview

`brain-cli` provides command-line interaction and developer ergonomics for the Brain ecosystem. It includes argument parsing, subcommands (train, eval, export, inspect, bench), an interactive mathematical REPL, shell completion generation (Bash, Zsh, Fish), ANSI terminal formatting, progress bars, and tensor summary inspection.

## Architecture

| Module | Description |
|---|---|
| `repl` | Interactive tensor & evaluation REPL with persistent session history |
| `commands` | Subcommand dispatch: `train`, `eval`, `export`, `benchmark`, `inspect` |
| `parser` | Custom zero-dependency flag and option CLI argument parser |
| `complete` | Auto-completion script generator for Bash, Zsh, and Fish |
| `ui` | Terminal tables, ANSI color themes, spinners, and progress meters |

## Quality & Verification

- **Tests**: 11,165 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-cli -- -D warnings`)
- **Dependencies**: `std` + `brain-core`
