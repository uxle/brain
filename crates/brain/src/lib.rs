//! Facade crate for the Brain framework.
//!
//! The facade re-exports the stable, currently wired parts of the workspace.
//! Domain crates can continue to evolve independently without forcing every
//! application to depend on every Brain crate.

/// Core tensor and utility APIs.
pub mod core {
    pub use brain_core::*;
}

#[cfg(feature = "autograd")]
/// Autograd APIs.
pub mod autograd {
    pub use brain_autograd::*;
}

#[cfg(feature = "data")]
/// Data loading APIs.
pub mod data {
    pub use brain_data::*;
    pub use brain_dataset as dataset;
}

#[cfg(feature = "export")]
/// Graph, ONNX, export, and quantization APIs.
pub mod export {
    pub use brain_export as artifact;
    pub use brain_graph as graph;
    pub use brain_onnx as onnx;
    pub use brain_quantization as quantization;
}

#[cfg(feature = "loss")]
/// Loss APIs.
pub mod loss {
    pub use brain_loss::*;
}

#[cfg(feature = "metric")]
/// Metric APIs.
pub mod metric {
    pub use brain_metric::*;
}

#[cfg(feature = "nn")]
/// Tensor-only neural-network module APIs.
pub mod nn {
    pub use brain_nn::*;
}

#[cfg(feature = "optim")]
/// Optimizer APIs.
pub mod optim {
    pub use brain_optim::*;
}

#[cfg(feature = "train")]
/// Integrated mutable training APIs.
pub mod train {
    pub use brain_train::*;
}

/// Common imports for Brain applications.
pub mod prelude {
    pub use brain_core::prelude::*;

    #[cfg(feature = "autograd")]
    pub use brain_autograd::prelude::*;

    #[cfg(feature = "loss")]
    pub use brain_loss::{ClassLossConfig, CrossEntropyLoss, Loss, LossKind, Reduction};

    #[cfg(feature = "metric")]
    pub use brain_metric::{Metric, MetricKind, MetricValue};

    #[cfg(feature = "optim")]
    pub use brain_optim::{
        optimizer::{Optimizer, ParamGroup, StepInfo},
        sgd::{Sgd, SgdConfig},
    };

    #[cfg(feature = "train")]
    pub use brain_train::prelude::*;
}

/// Returns the facade crate version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
