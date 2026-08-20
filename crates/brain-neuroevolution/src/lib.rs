//! # brain-neuroevolution
//!
//! Production-grade evolutionary computation, genetic algorithms, evolution strategies,
//! neuroevolution parameter mapping, and HyperNEAT for the Brain Framework.
//!
//! ## Architecture
//! - [`genome`] — `Genome` vector representation, random initializations, and encoding schemes
//! - [`population`] — Population management, elitism, and generation tracking
//! - [`selection`] — Tournament selection, Roulette-wheel, and Rank-based parent selection
//! - [`crossover`] — Single-point, Two-point, and Uniform crossover operators
//! - [`mutation`] — Gaussian perturbation with adaptive sigma and gene clamping
//! - [`fitness`] — `FitnessFn` objective evaluation trait and `FitnessStats` summary
//! - [`ga`] — Genetic Algorithm (`Ga`) engine and termination criteria
//! - [`es`] — Evolution Strategies (CMA-ES, (1+1)-ES with 1/5th success rule)
//! - [`neuro`] — Neural network weight encoding and layer parameter mapping
//! - [`hyperneat`] — HyperNEAT substrate geometry and Compositional Pattern-Producing Networks (CPPN)
//! - [`benchmark`] — Optimization benchmark functions (Sphere, Rosenbrock, Rastrigin, Ackley)
//! - [`parallel`] — Multi-threaded population evaluation via scoped threads
//! - [`checkpoint`] — Evolutionary snapshot creation and deterministic restoration
//! - [`core`] — `EvoConfig`, `AlgorithmKind`, `EvoError`, `EvoResult`
//! - [`ops`] — Tensor-genome flattening and parameter injection helpers
//! - [`utils`] — Fast deterministic RNG and fitness ranking

#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]

pub mod benchmark;
pub mod checkpoint;
pub mod config;
pub mod core;
pub mod crossover;
pub mod es;
pub mod fitness;
pub mod ga;
pub mod genome;
pub mod hyperneat;
pub mod impl_;
pub mod mutation;
pub mod neuro;
pub mod ops;
pub mod parallel;
pub mod population;
pub mod selection;
pub mod utils;

// ── Convenience re-exports ──────────────────────────────────────────────────
pub use benchmark::{ackley_fn, rastrigin_fn, rosenbrock_fn, sphere_fn};
pub use checkpoint::EvoCheckpoint;
pub use config::OperatorConfig;
pub use core::{AlgorithmKind, EvoConfig, EvoError, EvoResult};
pub use crossover::{
    blx_alpha_crossover, sbx_crossover, single_point_crossover, uniform_crossover,
};
pub use es::{Cmaes, CmaesConfig, Es1p1, Es1p1Config, EsKind, EsResult};
pub use fitness::{FitnessFn, FitnessStats};
pub use ga::{Ga, GaConfig, GaResult, TerminationConfig};
pub use genome::{EncodingKind, Genome, GenomeEncoding};
pub use hyperneat::{
    Cppn, CppnActivation, CppnNode, HyperneatConfig, SubstrateConfig, SubstrateGrid2D,
};
pub use impl_::run_evolution;
pub use mutation::mutate_gaussian;
pub use neuro::{
    flatten_layer_weights, total_neuro_parameters, unflatten_layer_weights, LayerWeightDescriptor,
    NeuroConfig,
};
pub use ops::{apply_to_weights, genome_to_tensor, tensor_to_genome};
pub use parallel::{evaluate_population_parallel, ParallelConfig};
pub use population::Population;
pub use selection::{rank_based_select, roulette_wheel_select, tournament_select, SelectionKind};
pub use utils::{rank_fitness, FastRng};

/// Framework version string.
pub const VERSION: &str = "0.2.0";
