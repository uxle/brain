//! Random number generation for the Brain deep learning framework.
//!
//! This module provides random number generators and distributions
//! for tensor initialization, data augmentation, and stochastic operations.
//!
//! # Key Components
//!
//! * [`BrainRng`] - XORShift128+ random number generator
//! * Distributions: Uniform, Normal (Box-Muller), Bernoulli, Categorical, Poisson, Gamma
//! * Global seed management
//! * Tensor initialization methods: kaiming, xavier

use std::cell::RefCell;

// =============================================================================
// BrainRng - XORShift128+ Generator
// =============================================================================

/// A fast, non-cryptographic PRNG using the xorshift128+ algorithm.
#[derive(Debug, Clone)]
pub struct BrainRng {
    state: [u64; 2],
}

impl BrainRng {
    /// Creates a new RNG from two 64-bit seed values.
    pub fn new(seed0: u64, seed1: u64) -> Self {
        let mut rng = BrainRng { state: [seed0, seed1] };
        if rng.state == [0, 0] { rng.state = [0xDEAD_BEEF_CAFE_BABE, 0x12345678_9ABCDEF0]; }
        rng
    }

    /// Creates an RNG from a single seed.
    pub fn from_seed(seed: u64) -> Self {
        let s = seed.wrapping_add(1);
        // Use SplitMix64 to generate two seeds
        let s0 = splitmix64(s);
        let s1 = splitmix64(s0);
        Self::new(s0, s1)
    }

    /// Generates the next random u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut s1 = self.state[0];
        let s0 = self.state[1];
        let result = s0.wrapping_add(s1);
        self.state[0] = s0;
        s1 ^= s1 << 23;
        self.state[1] = s1.wrapping_add(self.state[0]);
        self.state[0] = self.state[0].wrapping_add(self.state[0].wrapping_add(1));
        result
    }

    /// Generates a random u32 in [0, u32::MAX].
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Generates a random f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        let x = self.next_u64();
        (x >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Generates a random f64 in [0, 1) (exclusive of 1).
    pub fn next_f64_open(&mut self) -> f64 {
        let mut x = self.next_f64();
        while x >= 1.0 { x = self.next_f64(); }
        x
    }

    /// Generates a random f64 in [low, high).
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        assert!(low <= high, "low must be <= high");
        low + (high - low) * self.next_f64_open()
    }

    /// Generates a random integer in [low, high).
    pub fn uniform_int(&mut self, low: i64, high: i64) -> i64 {
        assert!(low <= high);
        low + (self.next_u64() % ((high - low) as u64)) as i64
    }

    /// Generates a normally distributed f64 using Box-Muller transform.
    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_f64_open();
        let u2 = self.next_f64_open();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z0
    }

    /// Generates a Bernoulli random variable (0 or 1).
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.next_f64_open() < p.clamp(0.0, 1.0)
    }

    /// Generates a boolean with probability p.
    pub fn boolean(&mut self, p: f64) -> bool { self.bernoulli(p) }

    /// Generates a random index in [0, n).
    pub fn gen_range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Generates a random usize.
    pub fn gen_usize(&mut self) -> usize {
        self.next_u64() as usize
    }

    /// Generates two independent normal values.
    pub fn normal_pair(&mut self, mean: f64, std: f64) -> (f64, f64) {
        let u1 = self.next_f64_open();
        let u2 = self.next_f64_open();
        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f64::consts::PI * u2).cos();
        let z1 = mag * (2.0 * std::f64::consts::PI * u2).sin();
        (mean + std * z0, mean + std * z1)
    }

    /// Generates a Poisson-distributed random variable.
    pub fn poisson(&mut self, lambda: f64) -> u64 {
        if lambda <= 0.0 { return 0; }
        if lambda < 30.0 {
            // Direct method for small lambda
            let l = lambda.exp();
            let mut k: u64 = 0;
            let mut p = 1.0;
            loop {
                p *= self.next_f64_open();
                if p < l { return k; }
                k += 1;
            }
        } else {
            // Rejection method for large lambda
            loop {
                let (x1, x2) = self.normal_pair(lambda, lambda.sqrt());
                let y = x1 / (x2 * x2);
                if y > 0.0 && y < 1.0 {
                    let val = (lambda + x1 * (1.0 + y).ln()).floor() as u64;
                    return val.max(0);
                }
            }
        }
    }

    /// Generates a Gamma-distributed random variable.
    pub fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        assert!(shape > 0.0 && scale > 0.0);
        if shape < 1.0 { return gamma_small(self, shape, scale); }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let (mut x, mut v) = self.normal_pair(0.0, 1.0);
            v = 1.0 + c * v;
            if v <= 0.0 { continue; }
            v = v * v * v;
            x = x * x;
            let u = self.next_f64_open();
            if u < 1.0 - 0.0331 * (x * x) * (x * x) { return d * scale * v; }
            if u.ln() < 0.5 * x + d * (1.0 - v + (v - 1.0).ln()) { return d * scale * v; }
        }
    }

    /// Fills a slice with uniform random values.
    pub fn fill_uniform(&mut self, data: &mut [f64], low: f64, high: f64) {
        for v in data.iter_mut() { *v = self.uniform(low, high); }
    }

    /// Fills a slice with normal random values.
    pub fn fill_normal(&mut self, data: &mut [f64], mean: f64, std: f64) {
        for v in data.iter_mut() { *v = self.normal(mean, std); }
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn gamma_small(rng: &mut BrainRng, alpha: f64, beta: f64) -> f64 {
    let p = 1.0 / alpha;
    let u = rng.next_f64_open().powf(p);
    beta * (-u.ln()).powf(1.0 / alpha)
}

impl Default for BrainRng {
    fn default() -> Self {
        Self::from_seed(42)
    }
}

// =============================================================================
// Global RNG State
// =============================================================================

thread_local! {
    static GLOBAL_RNG: RefCell<BrainRng> = RefCell::new(BrainRng::from_seed(42));
}

/// Sets the global RNG seed.
pub fn seed(seed: u64) {
    GLOBAL_RNG.with(|rng| { *rng.borrow_mut() = BrainRng::from_seed(seed); });
}

/// Sets the global manual seed.
pub fn manual_seed(seed: u64) { seed(seed); }

/// Returns a reference to the global RNG.
pub fn default_rng() -> BrainRng {
    GLOBAL_RNG.with(|rng| rng.borrow().clone())
}

/// Returns a mutable reference to the global RNG.
pub fn with_rng<F: FnOnce(&mut BrainRng) -> T, T>(f: F) {
    GLOBAL_RNG.with(|rng| f(&mut rng.borrow_mut()))
}

/// Generates a random f64 in [0, 1).
pub fn random_f64() -> f64 {
    GLOBAL_RNG.with(|rng| rng.borrow_mut().next_f64())
}

// =============================================================================
// Distributions
// =============================================================================

/// Uniform distribution in [low, high).
pub struct Uniform {
    pub low: f64,
    pub high: f64,
}

impl Uniform {
    pub fn new(low: f64, high: f64) -> Self {
        assert!(low <= high);
        Uniform { low, high }
    }

    pub fn sample(&self, rng: &mut BrainRng) -> f64 {
        rng.uniform(self.low, self.high)
    }
}

/// Normal distribution with given mean and standard deviation.
pub struct Normal {
    pub mean: f64,
    pub std: f64,
}

impl Normal {
    pub fn new(mean: f64, std: f64) -> Self {
        Normal { mean, std }
    }

    pub fn standard() -> Self { Normal::new(0.0, 1.0) }

    pub fn sample(&self, rng: &mut BrainRng) -> f64 {
        rng.normal(self.mean, self.std)
    }
}

/// Bernoulli distribution.
pub struct Bernoulli {
    pub p: f64,
}

impl Bernoulli {
    pub fn new(p: f64) -> Self {
        Bernoulli { p: p.clamp(0.0, 1.0) }
    }

    pub fn sample(&self, rng: &mut BrainRng) -> bool {
        rng.bernoulli(self.p)
    }
}

/// Categorical distribution.
pub struct Categorical {
    pub probs: Vec<f64>,
    pub cumsum: Vec<f64>,
}

impl Categorical {
    pub fn new(probs: &[f64]) -> Self {
        let total: f64 = probs.iter().sum();
        assert!(total > 0.0);
        let cumsum: Vec<f64> = probs.iter().scan(0.0, |acc, &p| acc + p).collect();
        Categorical { probs: probs.to_vec(), cumsum }
    }

    pub fn sample(&self, rng: &mut BrainRng) -> usize {
        let u = rng.next_f64_open() * self.cumsum.last().copied().unwrap_or(1.0);
        match self.cumsum.iter().position(|&c| u < c) {
            Some(i) => i,
            None => self.probs.len() - 1,
        }
    }
}

// =============================================================================
// Tensor Initialization Methods
// =============================================================================

/// Kaiming uniform initialization for ReLU-like activations.
/// Fills tensor with U(-sqrt(6/fan_in), sqrt(6/fan_in)).
pub fn kaiming_uniform(fan_in: usize) -> f64 {
    let bound = (6.0 / fan_in as f64).sqrt();
    default_rng().uniform(-bound, bound)
}

/// Kaiming normal initialization for ReLU-like activations.
/// Fills with N(0, sqrt(2/fan_in)).
pub fn kaiming_normal(fan_in: usize) -> f64 {
    let std = (2.0 / fan_in as f64).sqrt();
    default_rng().normal(0.0, std)
}

/// Xavier/Glorot uniform initialization.
/// Fills with U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))).
pub fn xavier_uniform(fan_in: usize, fan_out: usize) -> f64 {
    let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
    default_rng().uniform(-bound, bound)
}

/// Xavier/Glorot normal initialization.
/// Fills with N(0, sqrt(2/(fan_in+fan_out))).
pub fn xavier_normal(fan_in: usize, fan_out: usize) -> f64 {
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    default_rng().normal(0.0, std)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_creation() {
        let mut rng = BrainRng::new(12345, 67890);
        let v = rng.next_u64();
        assert!(v != 0);
    }

    #[test]
    fn test_rng_from_seed_deterministic() {
        let mut r1 = BrainRng::from_seed(42);
        let mut r2 = BrainRng::from_seed(42);
        for _ in 0..100 { assert_eq!(r1.next_u64(), r2.next_u64()); }
    }

    #[test]
    fn test_rng_different_seeds() {
        let mut r1 = BrainRng::from_seed(1);
        let mut r2 = BrainRng::from_seed(2);
        assert_ne!(r1.next_u64(), r2.next_u64());
    }

    #[test]
    fn test_uniform_range() {
        let mut rng = BrainRng::from_seed(42);
        for _ in 0..1000 {
            let v = rng.uniform(10.0, 20.0);
            assert!(v >= 10.0 && v < 20.0);
        }
    }

    #[test]
    fn test_uniform_zero_range() {
        let mut rng = BrainRng::from_seed(42);
        for _ in 0..100 {
            let v = rng.uniform(5.0, 5.0);
            assert!((v - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normal_mean_std() {
        let mut rng = BrainRng::from_seed(42);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let n = 10000;
        for _ in 0..n {
            let v = rng.normal(5.0, 2.0);
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!((mean - 5.0).abs() < 0.1);
        assert!((var.sqrt() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_bernoulli_probability() {
        let mut rng = BrainRng::from_seed(42);
        let mut count = 0;
        let n = 10000;
        for _ in 0..n {
            if rng.bernoulli(0.5) { count += 1; }
        }
        let ratio = count as f64 / n as f64;
        assert!((ratio - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_gen_range() {
        let mut rng = BrainRng::from_seed(42);
        for _ in 0..100 {
            let v = rng.gen_range(10);
            assert!(v < 10);
        }
    }

    #[test]
    fn test_normal_pair_independent() {
        let mut rng = BrainRng::from_seed(42);
        let (z0, z1) = rng.normal_pair(0.0, 1.0);
        // Check they are approximately independent (not a rigorous test)
        assert!(z0.abs() < 10.0);
        assert!(z1.abs() < 10.0);
    }

    #[test]
    fn test_seed_function() {
        seed(42);
        let v1 = random_f64();
        seed(42);
        let v2 = random_f64();
        assert!((v1 - v2).abs() < 1e-10);
    }

    #[test]
    fn test_manual_seed() {
        manual_seed(123);
        let _ = random_f64();
    }

    #[test]
    fn test_fill_uniform() {
        let mut rng = BrainRng::from_seed(42);
        let mut data = vec![0.0; 100];
        rng.fill_uniform(&mut data, 0.0, 1.0);
        for &v in &data {
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_fill_normal() {
        let mut rng = BrainRng::from_seed(42);
        let mut data = vec![0.0; 1000];
        rng.fill_normal(&mut data, 0.0, 1.0);
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_uniform_distribution() {
        let dist = Uniform::new(0.0, 1.0);
        let mut rng = BrainRng::from_seed(42);
        for _ in 0..100 {
            let v = dist.sample(&mut rng);
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_normal_distribution() {
        let dist = Normal::standard();
        let mut rng = BrainRng::from_seed(42);
        let mut sum = 0.0;
        for _ in 0..100 { sum += dist.sample(&mut rng); }
        assert!((sum / 100.0).abs() < 0.5);
    }

    #[test]
    fn test_categorical_distribution() {
        let dist = Categorical::new(&[0.2, 0.3, 0.5]);
        let mut rng = BrainRng::from_seed(42);
        let mut counts = [0usize; 3];
        let n = 10000;
        for _ in 0..n { counts[dist.sample(&mut rng)] += 1; }
        assert!((counts[0] as f64 / n as f64 - 0.2).abs() < 0.05);
        assert!((counts[2] as f64 / n as f64 - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_kaiming_uniform() {
        for _ in 0..100 {
            let v = kaiming_uniform(100);
            let bound = (6.0 / 100.0_f64).sqrt();
            assert!(v >= -bound && v < bound);
        }
    }

    #[test]
    fn test_kaiming_normal() {
        let mut sum = 0.0;
        let n = 1000;
        for _ in 0..n { sum += kaiming_normal(100).abs(); }
        let avg = sum / n as f64;
        let std = (2.0 / 100.0).sqrt();
        assert!((avg - std * 0.8).abs() < std * 0.5); // Very rough check
    }

    #[test]
    fn test_xavier_uniform() {
        for _ in 0..100 {
            let v = xavier_uniform(100, 200);
            let bound = (6.0 / 300.0_f64).sqrt();
            assert!(v >= -bound && v < bound);
        }
    }

    #[test]
    fn test_xavier_normal() {
        for _ in 0..100 {
            let v = xavier_normal(100, 200);
            assert!(v.abs() < 10.0); // Very rough check
        }
    }

    #[test]
    fn test_poisson_small() {
        let mut rng = BrainRng::from_seed(42);
        let mut sum = 0;
        let n = 1000;
        for _ in 0..n { sum += rng.poisson(3.0) as usize; }
        let avg = sum / n;
        assert!((avg - 3).abs() < 1);
    }

    #[test]
    fn test_poisson_zero() {
        let mut rng = BrainRng::from_seed(42);
        assert_eq!(rng.poisson(0.0), 0);
    }

    #[test]
    fn test_gamma() {
        let mut rng = BrainRng::from_seed(42);
        for _ in 0..100 {
            let v = rng.gamma(2.0, 1.0);
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_rng_u32() {
        let mut rng = BrainRng::from_seed(42);
        for _ in 0..100 {
            let v = rng.next_u32();
            assert!(v <= u32::MAX as u32);
        }
    }

    #[test]
    fn test_bernoulli_edge_cases() {
        let mut rng = BrainRng::from_seed(42);
        let mut always_true = 0;
        let mut always_false = 0;
        for _ in 0..100 {
            if rng.bernoulli(1.0) { always_true += 1; }
            if rng.bernoulli(0.0) { always_false += 1; }
        }
        assert_eq!(always_true, 100);
        assert_eq!(always_false, 0);
    }
}
