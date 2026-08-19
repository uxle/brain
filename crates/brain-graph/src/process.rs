//! # Pipeline Processor
//!
//! Multi-stage graph processing pipeline with per-pass verification hooks.
#![allow(missing_docs)]

use crate::core::GraphResult;
use crate::ir::{GraphIr, verify_graph};

/// Runs a custom graph transform with safety verification pre- and post-transform.
pub fn process_with_verification<F>(graph: &mut GraphIr, transform: F) -> GraphResult<()>
where
    F: FnOnce(&mut GraphIr) -> GraphResult<()>,
{
    verify_graph(graph)?;
    transform(graph)?;
    verify_graph(graph)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
