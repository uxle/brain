//! # Fault Tolerance & Heartbeat Monitoring
//!
//! Node failure detection, heartbeat pings, and automatic retry policies.

/// Fault handling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FaultPolicy {
    #[default]
    Retry,
    FailFast,
    ExcludeRank,
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
