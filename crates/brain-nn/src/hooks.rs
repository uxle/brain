//! # Module Execution Hooks
//!
//! Forward pre-hooks, forward post-hooks, and execution tracing registry.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Hook called before the forward pass of a module.
pub type ForwardPreHook = Box<dyn Fn(&Tensor) -> Tensor + Send + Sync>;

/// Hook called after the forward pass of a module.
pub type ForwardPostHook = Box<dyn Fn(&Tensor, &Tensor) -> Tensor + Send + Sync>;

/// Registry holding active forward hooks.
#[derive(Default)]
pub struct HookRegistry {
    pub pre_hooks: Vec<ForwardPreHook>,
    pub post_hooks: Vec<ForwardPostHook>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_pre_hook<F>(&mut self, hook: F)
    where
        F: Fn(&Tensor) -> Tensor + Send + Sync + 'static,
    {
        self.pre_hooks.push(Box::new(hook));
    }

    pub fn register_post_hook<F>(&mut self, hook: F)
    where
        F: Fn(&Tensor, &Tensor) -> Tensor + Send + Sync + 'static,
    {
        self.post_hooks.push(Box::new(hook));
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
