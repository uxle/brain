//! Tests for neuroevolution, genomes, and genetic operators
use brain_neuroevolution::*;
use brain_neuroevolution::utils::FastRng;

#[test]
fn test_genome_creation_and_mutation() {
    let mut rng = FastRng::seed(42);
    let mut g = Genome::random_uniform(5, -1.0, 1.0, &mut rng);
    assert_eq!(g.len(), 5);

    mutate_gaussian(&mut g, 0.5, 0.2, -5.0, 5.0, &mut rng);
    assert_eq!(g.len(), 5);
}

#[test]
fn test_benchmarks_and_evolution() {
    let val = sphere_fn(&[0.0, 0.0, 0.0]);
    assert_eq!(val, 0.0);
}
