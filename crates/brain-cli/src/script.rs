//! # Script Execution Mode & Shebang Runner
//!
//! Executes `.brain` procedural script files with variable scopes and status propagation.

use crate::core::{ExitCode, OutputSink};
use crate::repl::ReplState;

/// Executes a script from file contents line by line.
pub fn run_script(content: &str, sink: &OutputSink) -> ExitCode {
    let mut state = ReplState::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("#!") || trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }

        if let Err(err) = state.eval_line(trimmed, sink) {
            sink.println(&format!("Script error: {}", err));
            return ExitCode::ERROR;
        }
    }

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
