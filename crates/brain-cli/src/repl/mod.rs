//! # Interactive Mathematical REPL Session Engine
//!
//! Evaluates tensor expressions, manages variable bindings, and executes session commands.

pub mod completion;
pub mod parser;

use crate::core::{ExitCode, OutputSink};
use brain_core::Tensor;
use std::collections::HashMap;

/// REPL interactive session state and variable environment.
#[derive(Default)]
pub struct ReplState {
    pub variables: HashMap<String, Tensor>,
    pub history: Vec<String>,
}

impl ReplState {
    /// Creates a new `ReplState`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluates a single line of REPL input.
    pub fn eval_line(&mut self, line: &str, sink: &OutputSink) -> Result<(), String> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(());
        }

        self.history.push(trimmed.to_string());

        if trimmed.starts_with(':') {
            return self.handle_meta_command(trimmed, sink);
        }

        if let Some(eq_pos) = trimmed.find('=') {
            let var_name = trimmed[..eq_pos].trim().to_string();
            let expr = trimmed[eq_pos + 1..].trim();
            let val = parser::eval_expression(expr, &self.variables)?;
            sink.println(&format!("{} = {:?}", var_name, val.shape()));
            self.variables.insert(var_name, val);
        } else {
            let val = parser::eval_expression(trimmed, &self.variables)?;
            sink.println(&format!("ans: {:?}", val.shape()));
        }

        Ok(())
    }

    fn handle_meta_command(&mut self, cmd: &str, sink: &OutputSink) -> Result<(), String> {
        match cmd {
            ":help" => {
                sink.println("REPL Commands: :vars, :clear, :help, :quit, :exit");
            }
            ":vars" => {
                sink.println("Bound Variables:");
                for (k, v) in &self.variables {
                    sink.println(&format!("  {}: {:?}", k, v.shape()));
                }
            }
            ":clear" => {
                self.variables.clear();
                sink.println("Environment cleared.");
            }
            ":quit" | ":exit" => {
                sink.println("Goodbye!");
            }
            _ => {
                sink.println(&format!("Unknown REPL command: '{}'", cmd));
            }
        }
        Ok(())
    }
}

/// Runs the interactive computation REPL loop.
pub fn run_repl_command(sink: &OutputSink) -> ExitCode {
    sink.println("Brain Interactive REPL v0.2.0. Type ':help' for commands, ':quit' to exit.");
    let mut state = ReplState::new();
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        line.clear();
        match stdin.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed == ":quit" || trimmed == ":exit" {
                    sink.println("Goodbye!");
                    break;
                }
                if let Err(err) = state.eval_line(trimmed, sink) {
                    sink.println(&format!("error: {}", err));
                }
            }
            Err(_) => break,
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
