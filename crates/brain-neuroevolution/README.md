# `brain-neuroevolution`

Pure-Rust evolutionary computation: genetic algorithms, evolution strategies (CMA-ES, (1+1)-ES), neural weight encoding, and HyperNEAT.

## Overview

`brain-neuroevolution` provides gradient-free optimization and neuroevolution over `brain-core` tensors with zero external dependencies. It implements a complete GA engine (selection, crossover, mutation, elitism), CMA-ES and (1+1)-ES evolution strategies, tensor-to-genome weight mapping for neural networks, and HyperNEAT with CPPN-driven 2D substrates, plus parallel evaluation and checkpointing.

## Features

- **Genetic algorithm**: `Ga` engine (`GaConfig` = `EvoConfig`) with tournament selection, single-point/uniform crossover, Gaussian mutation, elitism, and `TerminationConfig`.
- **Evolution strategies**: `Cmaes` (`CmaesConfig`) and `Es1p1` (`Es1p1Config`) with `EsKind`/`EsResult`.
- **Neuroevolution**: `flatten_layer_weights`/`unflatten_layer_weights`/`total_neuro_parameters` with `LayerWeightDescriptor`; `genome_to_tensor`/`tensor_to_genome`/`apply_to_weights`.
- **HyperNEAT**: `Cppn` with `CppnNode`/`CppnActivation`, `SubstrateGrid2D`/`SubstrateConfig`, `HyperneatConfig`.
- **Population tooling**: `Population` management, `FitnessFn` trait + `FitnessStats`, `rank_fitness`, `FastRng`, and `evaluate_population_parallel`.
- **Benchmarks & checkpoints**: `sphere_fn`, `rosenbrock_fn`, `rastrigin_fn`, `ackley_fn`; `EvoCheckpoint` snapshots.
- **High-level runner**: `run_evolution(config, fitness_fn, seed)` returns best genome + fitness.

## Modules

| Module | Contents |
|---|---|
| `core`/`config` | `EvoConfig`, `AlgorithmKind`, `EvoError`/`EvoResult`, `OperatorConfig` |
| `genome`/`population`/`selection`/`crossover`/`mutation` | GA building blocks |
| `ga`/`es` | `Ga` engine + termination; `Cmaes`, `Es1p1` |
| `fitness` | `FitnessFn` trait, `FitnessStats` |
| `neuro`/`ops` | weight encoding/decoding, tensor↔genome conversion |
| `hyperneat` | `Cppn`, `SubstrateGrid2D`, configs |
| `benchmark`/`parallel`/`checkpoint`/`utils` | test functions, parallel evaluation, snapshots, RNG |

## Quick Start

```rust
use brain_neuroevolution::{run_evolution, EvoConfig, FitnessFn};

struct NegSphere;
impl FitnessFn for NegSphere {
    fn evaluate(&self, genes: &[f64]) -> f64 {
        -genes.iter().map(|g| g * g).sum::<f64>()
    }
}

let config = EvoConfig {
    population_size: 50,
    genome_dim: 5,
    max_generations: 100,
    ..Default::default()
};
let (best_genome, best_fitness) = run_evolution(&config, &NegSphere, 42).unwrap();
```

## Testing

```bash
cargo test -p brain-neuroevolution -j 2
```

## Workspace Role

Depends solely on `brain-core`; `brain-neuroevolution` provides gradient-free search and weight-injection strategies complementary to the autograd-based training path of the Brain framework.