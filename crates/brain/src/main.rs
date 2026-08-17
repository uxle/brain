//! # `brain` — the Brain deep-learning CLI binary.
//!
//! Thin entry point that forwards all command-line arguments to the
//! `brain-cli` library's dispatcher (`run_cli`). The heavy lifting (argument
//! parsing, subcommand routing, `make`/`run`, the REPL, diagnostics...) lives in
//! `brain-cli` so it stays independently testable.

use brain_cli::core::OutputSink;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sink = OutputSink::stdout();
    let exit_code = brain_cli::run_cli(&args, &sink);
    std::process::exit(exit_code.0);
}
