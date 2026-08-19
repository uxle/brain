//! # Dynamic Plugin & Subcommand Extensions
//!
//! Discovers external `brain-*` plugins and executes sidecar tools.

/// Plugin discovery registry.
#[derive(Default)]
pub struct PluginRegistry {
    discovered: Vec<String>,
}

impl PluginRegistry {
    /// Creates a new `PluginRegistry`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a plugin binary name.
    pub fn register(&mut self, name: impl Into<String>) {
        self.discovered.push(name.into());
    }

    /// Returns list of discovered plugin names.
    pub fn plugins(&self) -> &[String] {
        &self.discovered
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
