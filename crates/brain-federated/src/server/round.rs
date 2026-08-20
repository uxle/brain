//! # Round Lifecycle Management
//!
//! Manages the full select→distribute→collect→aggregate→evaluate cycle.
#![allow(missing_docs)]

use crate::core::RoundId;

/// Statistics gathered after a single federated training round.
#[derive(Debug, Clone, Default)]
pub struct RoundStats {
    pub round_id: RoundId,
    pub num_participants: usize,
    pub avg_loss: f64,
    pub duration_ms: u64,
}

impl RoundStats {
    pub fn new(round_id: RoundId, num_participants: usize) -> Self {
        Self {
            round_id,
            num_participants,
            avg_loss: 0.0,
            duration_ms: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
