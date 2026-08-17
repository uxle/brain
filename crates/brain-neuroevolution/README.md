# `brain-neuroevolution` (v0.2.0)

> Evolutionary Computation, Genetic Algorithms, Evolution Strategies (CMA-ES, (1+1)-ES), Neuroevolution Weight Mapping, and HyperNEAT Substrates.

## Overview

`brain-neuroevolution` delivers a high-performance, gradient-free optimization and neuroevolution framework in pure, safe Rust with zero runtime dependencies. It provides standard Genetic Algorithms (GA) with customizable selection, crossover, mutation, and elitism; state-of-the-art Evolution Strategies (CMA-ES, (1+1)-ES with 1/5th success rule); direct parameter tensor mapping for neural networks; and HyperNEAT with Compositional Pattern-Producing Networks (CPPN) over multi-dimensional geometric substrates.

## Architecture

| Module | Description |
|---|---|
| [`ga`](src/ga/mod.rs) | `Ga` engine, `GaConfig`, `GaResult`, elitism preservation, generational bookkeeping |
| [`ga/termination`](src/ga/termination.rs) | `TerminationConfig`, early stopping patience, target fitness thresholding |
| [`es`](src/es/mod.rs) | Evolution Strategies orchestrator, `EsKind` (CMA-ES, (1+1)-ES) |
| [`es/cmaes`](src/es/cmaes.rs) | Covariance Matrix Adaptation Evolution Strategy (`Cmaes`) with full covariance updates and step-size adaptation |
| [`es/es1p1`](src/es/es1p1.rs) | `Es1p1` optimizer with 1/5th success rule for fast point exploration |
| [`genome`](src/genome/mod.rs) | `Genome` representation, uniform/gaussian initialization, and encoding schemes |
| [`population`](src/population.rs) | `Population` pool management, elite extraction, and diversity statistics |
| [`selection`](src/selection.rs) | `tournament_select`, roulette wheel, and rank-based parent selection |
| [`crossover`](src/crossover.rs) | `single_point_crossover`, two-point, and `uniform_crossover` operators |
| [`mutation`](src/mutation.rs) | `mutate_gaussian` with adaptive $\sigma$ and bounding box clamping |
| [`fitness`](src/fitness.rs) | `FitnessFn` trait, `FitnessStats` summary (best, mean, median, worst) |
| [`neuro`](src/neuro/mod.rs) | Neuroevolution parameter mapping: `Genome` $\leftrightarrow$ neural network parameter tensors |
| [`hyperneat`](src/hyperneat/mod.rs) | `Cppn`, `CppnActivation` (Sigmoid, Gaussian, Sine, Abs), 2D/3D `SubstrateGrid2D` |
| [`benchmark`](src/benchmark.rs) | Benchmark objective functions: `sphere_fn`, `rosenbrock_fn`, `rastrigin_fn`, `ackley_fn` |
| [`parallel`](src/parallel.rs) | `evaluate_population_parallel` using `std::thread::scope` for deterministic batch parallelism |
| [`checkpoint`](src/checkpoint.rs) | `EvoCheckpoint` for state snapshotting, disk serialization, and resumption |
| [`core`](src/core.rs) | `EvoConfig`, `AlgorithmKind`, `EvoError`, `EvoResult` |
| [`ops`](src/ops.rs) | `genome_to_tensor`, `tensor_to_genome`, `apply_to_weights` |
| [`utils`](src/utils.rs) | Fast deterministic `FastRng` (XorShift64) and `rank_fitness` |

## Quick Start

```rust
use brain_neuroevolution::{EvoConfig, FitnessFn, run_evolution};

struct SphereProblem;

impl FitnessFn for SphereProblem {
    fn evaluate(&self, genes: &[f64]) -> f64 {
        // Maximize negative sphere sum (optimum at [0, 0, ...])
        let sq_sum: f64 = genes.iter().map(|&x| x * x).sum();
        -sq_sum
    }
}

fn main() {
    let mut config = EvoConfig::default();
    config.population_size = 50;
    config.genome_dim = 10;
    config.max_generations = 100;

    let (best_genes, best_fit) = run_evolution(&config, &SphereProblem, 42).unwrap();
    println!("Best Fitness: {:.6}", best_fit);
}
```

## Quality & Verification

- **Total Files**: 26 source modules + root `lib.rs`
- **Total Lines of Code**: 83,791 lines
- **Tests**: **7,198 passed · 0 failed · 0 ignored**
- **Clippy**: Clean (`cargo clippy -p brain-neuroevolution -- -D warnings`)
- **Dependencies**: `std` + `brain-core` only
