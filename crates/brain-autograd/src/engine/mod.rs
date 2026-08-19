//! # Autograd Execution Engines
//!
//! Advanced execution engines:
//! * [`parallel`] - Multi-threaded graph evaluation
//! * [`mixed`] - Mixed-precision scaling and stability guards

pub mod mixed;
pub mod parallel;

pub use mixed::GradScaler;
pub use parallel::{parallel_backward, ParallelConfig};
