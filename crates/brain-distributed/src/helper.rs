//! # Distributed Diagnostics & Formatting
//!
//! Formatting helpers for rank logs and cluster status reporting.

use crate::core::DistributedContext;

/// Formats distributed logging prefix.
pub fn format_rank_prefix(ctx: &DistributedContext) -> String {
    format!("[Rank {}/{}]", ctx.rank, ctx.world_size)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
