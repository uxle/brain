//! # Subcommand Registry & Execution Dispatcher
//!
//! Provides the [`Command`] trait, execution context, and command registry.

pub mod agent_cmd;
pub mod bench_cmd;
pub mod check_cmd;
pub mod convert_cmd;
pub mod dataset_cmd;
pub mod make_cmd;
pub mod model_cmd;
pub mod run_cmd;
pub mod space_cmd;
pub mod tensor_cmd;
pub mod train_cmd;

use crate::config::CliConfig;
use crate::core::{ExitCode, OutputSink};
use std::collections::HashMap;

/// Execution context provided to all command handlers.
pub struct CommandContext<'a> {
    pub config: &'a CliConfig,
    pub sink: &'a OutputSink,
    pub args: &'a [String],
}

/// Command handler trait for CLI subcommands.
pub trait Command: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn run(&self, ctx: &CommandContext) -> ExitCode;
}

/// Registry mapping command names to handlers.
#[derive(Default)]
pub struct CommandRegistry {
    commands: HashMap<String, Box<dyn Command>>,
}

impl CommandRegistry {
    /// Creates a new `CommandRegistry`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a command handler.
    pub fn register(&mut self, cmd: Box<dyn Command>) {
        self.commands.insert(cmd.name().to_string(), cmd);
    }

    /// Dispatches execution to the matching command handler.
    pub fn dispatch(&self, name: &str, ctx: &CommandContext) -> ExitCode {
        if let Some(cmd) = self.commands.get(name) {
            cmd.run(ctx)
        } else {
            ctx.sink.println(&format!("Command not found: '{}'", name));
            ExitCode::NOT_FOUND
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
