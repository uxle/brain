//! # Pipeline Error Handling & Recovery
//!
//! Error classifications distinguishing between retryable IO errors and fatal pipeline corruptions.

/// Diagnostic error conditions occurring during data ingestion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    IoError(String),
    CorruptSample(String),
    Timeout,
    WorkerDied,
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(s) => write!(f, "Pipeline IO error: {}", s),
            Self::CorruptSample(s) => write!(f, "Corrupt sample: {}", s),
            Self::Timeout => write!(f, "Pipeline operation timed out"),
            Self::WorkerDied => write!(f, "Worker thread terminated unexpectedly"),
        }
    }
}

impl std::error::Error for PipelineError {}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
