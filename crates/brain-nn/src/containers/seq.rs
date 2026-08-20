//! # Sequential Execution Container
//!
//! Standard sequential pipeline forwarding tensors through submodules in insertion order.
#![allow(missing_docs)]

use crate::module::{Module, ModuleResult};

/// Sequential container chaining modules in forward order.
pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

use brain_autograd::Value;

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add<M: Module + 'static>(&mut self, module: M) {
        self.layers.push(Box::new(module));
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let mut cur = input.clone();
        for layer in &self.layers {
            cur = layer.forward(&cur)?;
        }
        Ok(cur)
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
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
