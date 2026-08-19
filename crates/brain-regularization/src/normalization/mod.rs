//! # Normalization Family Modules
//!
//! BatchNorm (1D/2D/3D), LayerNorm, RMSNorm, GroupNorm, InstanceNorm, and Weight/Spectral Normalization.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

pub mod batch;
pub mod layer;
pub mod group;
pub mod weight;

pub use batch::{BatchNorm1d, BatchNorm2d, BatchNorm3d, BatchNormConfig};
pub use layer::{LayerNorm, LayerNormConfig, RMSNorm};
pub use group::{GroupNorm, GroupNormConfig, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, InstanceNormConfig};
pub use weight::{SpectralNorm, SpectralNormConfig, WeightNorm};

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::dropout::*;
    use crate::normalization::*;
    use crate::regularizers::*;
    use crate::decay::*;
    use crate::earlystop::*;
    use crate::stopping::*;
    use crate::augment::*;
    use crate::perturb::*;
    use crate::dropout_uncertainty::*;
    use crate::label_smooth::*;
    use crate::curriculum::*;
    use crate::consistency::*;
    use crate::rules::*;
    use crate::registry::*;
    use crate::train_hooks::*;
    use crate::ops::*;
    use crate::r#impl::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
