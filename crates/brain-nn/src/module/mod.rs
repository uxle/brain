//! # Module System & Layer Architecture
//!
//! Master `Module` trait, parameter/buffer discovery, training/evaluation modes, and container lists.
#![allow(missing_docs)]

pub mod parameter;
pub use parameter::{Parameter, Buffer, NamedParameter};

use brain_core::Tensor;
use std::collections::HashMap;

/// Error type for neural network modules.
#[derive(Debug, Clone, PartialEq)]
pub enum ModuleError {
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    InvalidParameter(String),
    MissingState(String),
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModuleError::ShapeMismatch { expected, got } => write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got),
            ModuleError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            ModuleError::MissingState(msg) => write!(f, "Missing state: {}", msg),
        }
    }
}

pub type ModuleResult<T> = Result<T, ModuleError>;

/// Core trait representing a neural network module or layer.
pub trait Module: Send + Sync {
    /// Executes the forward pass of the module.
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor>;

    /// Returns a list of all trainable parameter tensors.
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }

    /// Sets module training mode (true for training, false for evaluation).
    fn set_training(&mut self, _training: bool) {}

    /// Exports all named parameter states into a dictionary.
    fn state_dict(&self) -> HashMap<String, Tensor> {
        HashMap::new()
    }

    /// Loads parameter states from a dictionary.
    fn load_state_dict(&mut self, _state: &HashMap<String, Tensor>) -> ModuleResult<()> {
        Ok(())
    }
}

/// Sequential list of sub-modules executed in order.
pub struct ModuleList {
    pub modules: Vec<Box<dyn Module>>,
}

impl ModuleList {
    pub fn new() -> Self {
        Self { modules: Vec::new() }
    }

    pub fn add<M: Module + 'static>(&mut self, module: M) {
        self.modules.push(Box::new(module));
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl Module for ModuleList {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        let mut cur = input.clone();
        for m in &self.modules {
            cur = m.forward(&cur)?;
        }
        Ok(cur)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }
}

/// Ordered key-value dictionary of sub-modules.
pub struct ModuleDict {
    pub modules: std::collections::BTreeMap<String, Box<dyn Module>>,
}

impl ModuleDict {
    pub fn new() -> Self {
        Self { modules: std::collections::BTreeMap::new() }
    }

    pub fn insert<M: Module + 'static>(&mut self, key: impl Into<String>, module: M) {
        self.modules.insert(key.into(), Box::new(module));
    }

    pub fn get(&self, key: &str) -> Option<&Box<dyn Module>> {
        self.modules.get(key)
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl Default for ModuleDict {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ModuleDict {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.modules.values().flat_map(|m| m.parameters()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Linear;

    #[test]
    fn test_container_parameter_completeness() {
        let mut seq = ModuleList::new();
        seq.add(Linear::new(4, 8, true));  // weight [8, 4] (32), bias [8] (8) -> 40
        seq.add(Linear::new(8, 2, false)); // weight [2, 8] (16), no bias -> 16
        let params = seq.parameters();
        assert_eq!(params.len(), 3);
        let total_elems: usize = params.iter().map(|p| p.numel()).sum();
        assert_eq!(total_elems, 56);

        let mut dict = ModuleDict::new();
        dict.insert("l1", Linear::new(4, 8, true));
        dict.insert("l2", Linear::new(8, 2, false));
        let dict_params = dict.parameters();
        assert_eq!(dict_params.len(), 3);
        let dict_elems: usize = dict_params.iter().map(|p| p.numel()).sum();
        assert_eq!(dict_elems, 56);
    }
}
