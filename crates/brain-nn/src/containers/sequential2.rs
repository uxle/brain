//! # Extended Sequential Container
//!
//! Named sequential execution pipeline with forward hook dispatch and layer indexing.
#![allow(missing_docs)]

use crate::module::{Module, ModuleResult};

/// Named layer entry in an extended sequential container.
pub struct NamedModule {
    pub name: String,
    pub module: Box<dyn Module>,
}

/// Extended sequential container maintaining named child modules.
pub struct SequentialNamed {
    pub children: Vec<NamedModule>,
}

use brain_autograd::Value;

impl SequentialNamed {
    pub fn new() -> Self {
        Self {
            children: Vec::new(),
        }
    }

    pub fn add<M: Module + 'static>(&mut self, name: impl Into<String>, module: M) {
        self.children.push(NamedModule {
            name: name.into(),
            module: Box::new(module),
        });
    }

    pub fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let mut cur = input.clone();
        for child in &self.children {
            cur = child.module.forward(&cur)?;
        }
        Ok(cur)
    }
}

impl Default for SequentialNamed {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SequentialNamed {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Value> {
        self.children
            .iter()
            .flat_map(|c| c.module.parameters())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
