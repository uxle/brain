//! Tests for neuroevolution, genomes, and genetic operators
use brain_neuroevolution::utils::FastRng;
use brain_neuroevolution::*;

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

#[test]
fn test_advanced_crossover_and_selection() {
    let mut rng = FastRng::seed(100);
    let p1 = Genome::new(vec![1.0, 5.0]);
    let p2 = Genome::new(vec![3.0, 15.0]);

    let (c1, c2) = blx_alpha_crossover(&p1, &p2, 0.2, &mut rng);
    assert_eq!(c1.len(), 2);
    assert_eq!(c2.len(), 2);

    let (s1, s2) = sbx_crossover(&p1, &p2, 2.0, &mut rng);
    assert_eq!(s1.len(), 2);
    assert_eq!(s2.len(), 2);

    let mut g1 = Genome::new(vec![0.0]);
    g1.fitness = Some(10.0);
    let mut g2 = Genome::new(vec![1.0]);
    g2.fitness = Some(50.0);

    let pop = vec![g1, g2];
    let selected_roulette = roulette_wheel_select(&pop, &mut rng);
    assert!(selected_roulette.fitness.is_some());

    let selected_rank = rank_based_select(&pop, &mut rng);
    assert!(selected_rank.fitness.is_some());
}
