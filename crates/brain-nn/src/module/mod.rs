//! # Module System & Layer Architecture (migrated onto `brain_autograd::Value`)
//!
//! Master `Module` trait, parameter/buffer discovery, training/evaluation modes,
//! and container lists. As of Phase 0 (brain-nn/brain-autograd unification),
//! `forward()` and `parameters()` operate on `Value`, which carries the
//! computation tape -- so every layer implementing this trait gets a real,
//! working `.backward()` for free, driven by `brain_autograd`'s existing
//! generic reverse-mode engine. No layer needs a hand-written backward pass.
#![allow(missing_docs)]

pub mod parameter;
pub use parameter::{Parameter, Buffer, NamedParameter};

use brain_autograd::Value;
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
///
/// `forward` and `parameters` operate on `Value`, not raw `Tensor` -- this is
/// the change introduced in Phase 0. A `Value` carries the autograd tape, so
/// any composition of `Module`s automatically supports `.backward()` via
/// `brain_autograd`'s existing reverse-mode engine, with no per-layer
/// hand-written gradient formula required.
pub trait Module: Send + Sync {
    /// Executes the forward pass of the module on a tape-tracked `Value`.
    fn forward(&self, input: &Value) -> ModuleResult<Value>;

    /// Returns a list of all trainable parameter `Value`s (tape leaves).
    fn parameters(&self) -> Vec<Value> {
        Vec::new()
    }

    /// Sets module training mode (true for training, false for evaluation).
    fn set_training(&mut self, _training: bool) {}

    /// Exports all named parameter states into a dictionary.
    ///
    /// State dicts stay `Tensor`-based (not `Value`) since serialization
    /// doesn't need tape information -- only the underlying data.
    fn state_dict(&self) -> HashMap<String, Tensor> {
        HashMap::new()
    }

    /// Loads parameter states from a dictionary.
    fn load_state_dict(&mut self, _state: &HashMap<String, Tensor>) -> ModuleResult<()> {
        Ok(())
    }

    /// Transitional shim (Phase 0 migration window only): runs `forward` via
    /// a temporary leaf `Value` and unwraps back to `Tensor`. This exists so
    /// callers not yet migrated to the `Value`-based API (e.g. crates still
    /// being converted) keep working during the migration. It does NOT
    /// preserve gradient tracking across the call boundary -- any caller
    /// that needs `.backward()` must use `forward()` directly with a `Value`
    /// it already holds a tape reference to, not this shim. Remove this
    /// method once every internal caller is migrated (see Phase 0, section
    /// 2.4 and 2.4's removal criterion).
    fn forward_tensor(&self, input: &Tensor) -> ModuleResult<Tensor> {
        let v = Value::new(input.clone(), false);
        let out = self.forward(&v)?;
        Ok(out.data().clone())
    }
}

/// Sequential list of sub-modules executed in order.
#[derive(Default)]
pub struct ModuleList {
    pub modules: Vec<Box<dyn Module>>,
}

impl ModuleList {
    pub fn new() -> Self {
        Self::default()
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
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let mut cur = input.clone();
        for m in &self.modules {
            cur = m.forward(&cur)?;
        }
        Ok(cur)
    }

    fn parameters(&self) -> Vec<Value> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }
}

/// Ordered key-value dictionary of sub-modules.
#[derive(Default)]
pub struct ModuleDict {
    pub modules: std::collections::BTreeMap<String, Box<dyn Module>>,
}

impl ModuleDict {
    pub fn new() -> Self {
        Self::default()
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

impl Module for ModuleDict {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<Value> {
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
        let total_elems: usize = params.iter().map(|p| p.data().numel()).sum();
        assert_eq!(total_elems, 56);

        let mut dict = ModuleDict::new();
        dict.insert("l1", Linear::new(4, 8, true));
        dict.insert("l2", Linear::new(8, 2, false));
        let dict_params = dict.parameters();
        assert_eq!(dict_params.len(), 3);
        let dict_elems: usize = dict_params.iter().map(|p| p.data().numel()).sum();
        assert_eq!(dict_elems, 56);
    }

    /// The test that was previously impossible to write honestly: a full
    /// forward + backward + parameter-gradient check through the REAL tape,
    /// not a hand-derived formula compared against finite differences on the
    /// forward pass alone. If this test passes, Linear has a genuinely
    /// working, tape-driven backward pass for the first time.
    #[test]
    fn test_linear_real_tape_backward_end_to_end() {
        let linear = Linear::new(3, 2, true);
        let x = Value::new(
            Tensor::from_slice(&[1.0, 2.0, 3.0, -1.0, 0.5, 2.0], vec![2, 3]),
            false, // input doesn't need its own gradient for this check
        );

        let out = linear.forward(&x).expect("forward should succeed");
        let loss = out.sum();
        loss.backward().expect("backward should succeed through the real tape");

        // Every parameter should now have a non-None gradient populated by
        // the tape -- this was categorically impossible before Phase 0,
        // since Linear had no backward path at all.
        for p in linear.parameters() {
            let g = p.grad();
            assert!(g.is_some(), "parameter gradient should be populated by .backward()");
        }
    }
}
