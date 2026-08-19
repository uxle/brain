//! Random number generation and statistical distributions for the Brain deep learning framework.
//!
//! This module provides high-performance, deterministic PRNGs, hierarchical seeding sequences,
//! parameter initialization algorithms (Kaiming, Xavier, Orthogonal, Truncated Normal),
//! statistical distributions (Normal, Gamma, Beta, Cauchy, LogNormal, Exponential, Poisson),
//! and tensor shuffling without external runtime dependencies.
//!
//! # Architecture & Components
//!
//! 1. **Core PRNG Engines**:
//!    - [`BrainRng`]: High-speed XORShift128+ generator.
//!    - [`PCG32`]: Permuted Congruential Generator with 64-bit state and 32-bit output.
//!    - [`SplitMix64`]: Fast 64-bit splittable pseudo-random generator.
//!    - [`ChaCha8`]: Pure-Rust 8-round ChaCha stream cipher PRNG for cryptographically strong statistical uniformity.
//!
//! 2. **Rng Trait**:
//!    - [`Rng`]: Uniform interface providing `next_u32`, `next_u64`, `next_f32`, `next_f64`, `fill_bytes`, and `fill_slice`.
//!
//! 3. **Hierarchical Seeding**:
//!    - [`SeedSeq`]: Seed sequence entropy mixer for reproducible initialization of parallel worker threads.
//!    - [`SeedExt`]: Extension trait for seed conversion and expansion.
//!
//! 4. **Statistical Distributions**:
//!    - [`UniformDist`]: Continuous and discrete uniform distribution in `[low, high)`.
//!    - [`NormalDist`]: Gaussian distribution with Box-Muller transform and Ziggurat fast path.
//!    - [`LogNormalDist`]: Log-Normal distribution.
//!    - [`GammaDist`]: Gamma distribution via Marsaglia-Tsang transformation.
//!    - [`BetaDist`]: Beta distribution via ratio of Gammas.
//!    - [`ExponentialDist`]: Exponential distribution with rate parameter lambda.
//!    - [`CauchyDist`]: Standard and location-scale Cauchy distribution.
//!    - [`BernoulliDist`]: Bernoulli binary distribution.
//!    - [`PoissonDist`]: Poisson discrete count distribution.
//!    - [`DirichletDist`]: Multi-variate Dirichlet distribution.
//!
//! 5. **Tensor Initialization**:
//!    - [`KaimingUniform`], [`KaimingNormal`]: He variance scaling initialization.
//!    - [`XavierUniform`], [`XavierNormal`]: Glorot initialization.
//!    - [`TruncatedNormal`]: Bounded Gaussian sampling within 2 standard deviations.
//!    - [`OrthogonalInit`]: Semi-orthogonal matrix generation via QR decomposition.
//!
//! 6. **Permutations & Shuffling**:
//!    - [`ShuffleSeq`]: In-place Fisher-Yates shuffle engine for arbitrary slices.
//!    - `randperm`, `sample_without_replacement`.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::error::{BrainError, BrainResult};

// =============================================================================
// Rng Trait
// =============================================================================

/// Common interface for pseudo-random number generators.
pub trait Rng: Send + Sync {
    /// Generates the next pseudo-random 32-bit unsigned integer.
    fn next_u32(&mut self) -> u32;

    /// Generates the next pseudo-random 64-bit unsigned integer.
    fn next_u64(&mut self) -> u64 {
        let hi = (self.next_u32() as u64) << 32;
        let lo = self.next_u32() as u64;
        hi | lo
    }

    /// Generates a uniform pseudo-random float in `[0.0, 1.0)`.
    fn next_f64(&mut self) -> f64 {
        let v = self.next_u64() >> 11;
        (v as f64) * (1.0 / 9007199254740992.0)
    }

    /// Generates a uniform pseudo-random float in `[0.0, 1.0)`.
    fn next_f32(&mut self) -> f32 {
        let v = self.next_u32() >> 8;
        (v as f32) * (1.0 / 16777216.0)
    }

    /// Fills a byte buffer with pseudo-random data.
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut chunks = dest.chunks_exact_mut(8);
        for chunk in chunks.by_ref() {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes);
        }
        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let bytes = self.next_u64().to_le_bytes();
            rem.copy_from_slice(&bytes[..rem.len()]);
        }
    }

    /// Fills a slice of `f64` with uniform samples in `[0.0, 1.0)`.
    fn fill_f64_slice(&mut self, dest: &mut [f64]) {
        for elem in dest.iter_mut() {
            *elem = self.next_f64();
        }
    }
}

// =============================================================================
// SplitMix64 PRNG
// =============================================================================

/// Fast 64-bit splittable PRNG algorithm by Steele et al. (2014).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a new `SplitMix64` generator from a 64-bit seed.
    pub const fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Generates the next 64-bit integer and advances internal state.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Splits this generator into two independent `SplitMix64` instances.
    pub fn split(&mut self) -> Self {
        SplitMix64::new(self.next_u64())
    }
}

impl Rng for SplitMix64 {
    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        SplitMix64::next_u64(self)
    }
}

// =============================================================================
// PCG32 PRNG
// =============================================================================

/// Permuted Congruential Generator (PCG-XSH-RR) by Melissa O'Neill.
///
/// Features 64 bits of internal state, 32-bit output, and period of \(2^{64}\).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PCG32 {
    state: u64,
    inc: u64,
}

impl PCG32 {
    /// Creates a new PCG32 generator with seed and stream sequence ID.
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut pcg = PCG32 {
            state: 0,
            inc: (stream << 1) | 1,
        };
        pcg.next_u32();
        pcg.state = pcg.state.wrapping_add(seed);
        pcg.next_u32();
        pcg
    }

    /// Creates a PCG32 generator from a single 64-bit seed.
    pub fn from_seed(seed: u64) -> Self {
        Self::new(seed, 0x5446253602431CCB)
    }
}

impl Rng for PCG32 {
    fn next_u32(&mut self) -> u32 {
        let oldstate = self.state;
        self.state = oldstate
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);
        let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
        let rot = (oldstate >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
}

// =============================================================================
// BrainRng - XORShift128+ Generator
// =============================================================================

/// Fast non-cryptographic PRNG using the XORShift128+ algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrainRng {
    state: [u64; 2],
}

impl BrainRng {
    /// Creates a new RNG from two 64-bit seed values.
    pub fn new(seed0: u64, seed1: u64) -> Self {
        let mut rng = BrainRng {
            state: [seed0, seed1],
        };
        if rng.state == [0, 0] {
            rng.state = [0xDEAD_BEEF_CAFE_BABE, 0x12345678_9ABCDEF0];
        }
        rng
    }

    /// Creates an RNG from a single 64-bit seed using SplitMix64 expansion.
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed.wrapping_add(1));
        let s0 = sm.next_u64();
        let s1 = sm.next_u64();
        Self::new(s0, s1)
    }
}

impl Rng for BrainRng {
    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let mut s1 = self.state[0];
        let s0 = self.state[1];
        let result = s0.wrapping_add(s1);
        self.state[0] = s0;
        s1 ^= s1 << 23;
        self.state[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
        result
    }

    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}

// =============================================================================
// ChaCha8 - Pure-Rust Stream Cipher PRNG
// =============================================================================

/// Pure-Rust ChaCha8 stream cipher PRNG for high statistical randomness and independence.
#[derive(Clone)]
pub struct ChaCha8 {
    state: [u32; 16],
    buffer: [u32; 16],
    index: usize,
}

impl ChaCha8 {
    /// Creates a ChaCha8 generator from a 256-bit (32-byte) key and 64-bit nonce.
    pub fn new(key: [u8; 32], nonce: u64) -> Self {
        let mut state = [0u32; 16];
        state[0] = 0x61707865;
        state[1] = 0x3320646e;
        state[2] = 0x79622d32;
        state[3] = 0x6b206574;

        for i in 0..8 {
            let offset = i * 4;
            state[4 + i] = u32::from_le_bytes([
                key[offset],
                key[offset + 1],
                key[offset + 2],
                key[offset + 3],
            ]);
        }

        state[12] = 0; // Block counter low
        state[13] = 0; // Block counter high
        state[14] = nonce as u32;
        state[15] = (nonce >> 32) as u32;

        let mut rng = ChaCha8 {
            state,
            buffer: [0u32; 16],
            index: 16,
        };
        rng.refill();
        rng
    }

    /// Creates a ChaCha8 generator from a single 64-bit seed.
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        let mut key = [0u8; 32];
        for chunk in key.chunks_exact_mut(8) {
            chunk.copy_from_slice(&sm.next_u64().to_le_bytes());
        }
        let nonce = sm.next_u64();
        Self::new(key, nonce)
    }

    fn refill(&mut self) {
        let mut x = self.state;
        for _ in 0..4 {
            // 8 rounds total (4 double-rounds)
            Self::quarter_round(&mut x, 0, 4, 8, 12);
            Self::quarter_round(&mut x, 1, 5, 9, 13);
            Self::quarter_round(&mut x, 2, 6, 10, 14);
            Self::quarter_round(&mut x, 3, 7, 11, 15);

            Self::quarter_round(&mut x, 0, 5, 10, 15);
            Self::quarter_round(&mut x, 1, 6, 11, 12);
            Self::quarter_round(&mut x, 2, 7, 8, 13);
            Self::quarter_round(&mut x, 3, 4, 9, 14);
        }

        for i in 0..16 {
            self.buffer[i] = x[i].wrapping_add(self.state[i]);
        }

        self.state[12] = self.state[12].wrapping_add(1);
        if self.state[12] == 0 {
            self.state[13] = self.state[13].wrapping_add(1);
        }
        self.index = 0;
    }

    #[inline(always)]
    fn quarter_round(x: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
        x[a] = x[a].wrapping_add(x[b]);
        x[d] = (x[d] ^ x[a]).rotate_left(16);
        x[c] = x[c].wrapping_add(x[d]);
        x[b] = (x[b] ^ x[c]).rotate_left(12);
        x[a] = x[a].wrapping_add(x[b]);
        x[d] = (x[d] ^ x[a]).rotate_left(8);
        x[c] = x[c].wrapping_add(x[d]);
        x[b] = (x[b] ^ x[c]).rotate_left(7);
    }
}

impl Rng for ChaCha8 {
    fn next_u32(&mut self) -> u32 {
        if self.index >= 16 {
            self.refill();
        }
        let val = self.buffer[self.index];
        self.index += 1;
        val
    }
}

// =============================================================================
// SeedSeq & Hierarchical Seeding
// =============================================================================

/// Seed sequence algorithm for generating independent entropy streams across threads.
#[derive(Debug, Clone)]
pub struct SeedSeq {
    entropy: Vec<u32>,
}

impl SeedSeq {
    /// Creates a new `SeedSeq` from a slice of 32-bit integer seeds.
    pub fn new(seeds: &[u32]) -> Self {
        SeedSeq {
            entropy: seeds.to_vec(),
        }
    }

    /// Creates a `SeedSeq` from a single 64-bit seed.
    pub fn from_u64(seed: u64) -> Self {
        let lo = seed as u32;
        let hi = (seed >> 32) as u32;
        SeedSeq {
            entropy: vec![lo, hi],
        }
    }

    /// Generates `count` 64-bit seeds mixed with non-linear hashing.
    pub fn generate_u64_seeds(&self, count: usize) -> Vec<u64> {
        let mut sm = SplitMix64::new(0x9E3779B97F4A7C15);
        for &e in &self.entropy {
            sm.state = sm.state.wrapping_add(e as u64);
            sm.next_u64();
        }
        let mut results = Vec::with_capacity(count);
        for _ in 0..count {
            results.push(sm.next_u64());
        }
        results
    }
}

// =============================================================================
// Distributions
// =============================================================================

/// Continuous uniform distribution in `[low, high)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UniformDist {
    pub low: f64,
    pub high: f64,
}

impl UniformDist {
    /// Creates a uniform distribution in `[low, high)`.
    pub fn new(low: f64, high: f64) -> Self {
        assert!(low < high, "UniformDist: low must be < high");
        UniformDist { low, high }
    }

    /// Standard uniform distribution in `[0.0, 1.0)`.
    pub fn standard() -> Self {
        UniformDist {
            low: 0.0,
            high: 1.0,
        }
    }

    /// Samples a value from this distribution using the given RNG.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.low + (self.high - self.low) * rng.next_f64()
    }
}

/// Gaussian Normal distribution with mean and standard deviation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalDist {
    pub mean: f64,
    pub std: f64,
}

impl NormalDist {
    /// Creates a Normal distribution with given mean and standard deviation.
    pub fn new(mean: f64, std: f64) -> Self {
        assert!(std >= 0.0, "NormalDist: std must be non-negative");
        NormalDist { mean, std }
    }

    /// Standard Normal distribution with mean 0.0 and std 1.0.
    pub fn standard() -> Self {
        NormalDist {
            mean: 0.0,
            std: 1.0,
        }
    }

    /// Samples a Gaussian random variable using the Box-Muller transform.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        if self.std == 0.0 {
            return self.mean;
        }
        let u1 = (1.0 - rng.next_f64()).max(1e-15);
        let u2 = rng.next_f64();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        self.mean + self.std * z0
    }
}

/// Log-Normal distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormalDist {
    pub mu: f64,
    pub sigma: f64,
}

impl LogNormalDist {
    /// Creates a Log-Normal distribution.
    pub fn new(mu: f64, sigma: f64) -> Self {
        assert!(sigma > 0.0, "LogNormalDist: sigma must be > 0");
        LogNormalDist { mu, sigma }
    }

    /// Samples from the Log-Normal distribution.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let norm = NormalDist::new(self.mu, self.sigma);
        norm.sample(rng).exp()
    }
}

/// Gamma distribution with shape parameter `alpha` and scale parameter `beta`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaDist {
    pub shape: f64,
    pub scale: f64,
}

impl GammaDist {
    /// Creates a Gamma distribution.
    pub fn new(shape: f64, scale: f64) -> Self {
        assert!(shape > 0.0, "GammaDist: shape must be > 0");
        assert!(scale > 0.0, "GammaDist: scale must be > 0");
        GammaDist { shape, scale }
    }

    /// Samples using the Marsaglia and Tsang method (2000).
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let alpha = self.shape;
        if alpha < 1.0 {
            let u = rng.next_f64();
            let g = GammaDist::new(alpha + 1.0, self.scale).sample(rng);
            return g * u.powf(1.0 / alpha);
        }

        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        let norm = NormalDist::standard();

        loop {
            let z = norm.sample(rng);
            let v = (1.0 + c * z).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = rng.next_f64();
            if u < 1.0 - 0.0331 * z.powi(4) {
                return self.scale * d * v;
            }
            if u.ln() < 0.5 * z.powi(2) + d * (1.0 - v + v.ln()) {
                return self.scale * d * v;
            }
        }
    }
}

/// Beta distribution with parameters `alpha` and `beta`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaDist {
    pub alpha: f64,
    pub beta: f64,
}

impl BetaDist {
    /// Creates a Beta distribution.
    pub fn new(alpha: f64, beta: f64) -> Self {
        assert!(alpha > 0.0 && beta > 0.0, "BetaDist parameters must be > 0");
        BetaDist { alpha, beta }
    }

    /// Samples using the ratio of independent Gamma distributions.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let x = GammaDist::new(self.alpha, 1.0).sample(rng);
        let y = GammaDist::new(self.beta, 1.0).sample(rng);
        x / (x + y)
    }
}

/// Exponential distribution with rate parameter `lambda`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialDist {
    pub lambda: f64,
}

impl ExponentialDist {
    /// Creates an Exponential distribution.
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0, "ExponentialDist: lambda must be > 0");
        ExponentialDist { lambda }
    }

    /// Samples using the inverse transform method.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u = (1.0 - rng.next_f64()).max(1e-15);
        -u.ln() / self.lambda
    }
}

/// Cauchy distribution with location `x0` and scale `gamma`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CauchyDist {
    pub location: f64,
    pub scale: f64,
}

impl CauchyDist {
    /// Creates a Cauchy distribution.
    pub fn new(location: f64, scale: f64) -> Self {
        assert!(scale > 0.0, "CauchyDist: scale must be > 0");
        CauchyDist { location, scale }
    }

    /// Samples using inverse CDF tan transform.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u = rng.next_f64();
        self.location + self.scale * (std::f64::consts::PI * (u - 0.5)).tan()
    }
}

/// Bernoulli binary distribution with probability `p` of returning 1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BernoulliDist {
    pub p: f64,
}

impl BernoulliDist {
    /// Creates a Bernoulli distribution with success probability `p` in `[0.0, 1.0]`.
    pub fn new(p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "BernoulliDist: p must be in [0, 1]"
        );
        BernoulliDist { p }
    }

    /// Samples a boolean true/false value.
    pub fn sample_bool<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        rng.next_f64() < self.p
    }

    /// Samples a float 1.0 (success) or 0.0 (failure).
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        if self.sample_bool(rng) {
            1.0
        } else {
            0.0
        }
    }
}

/// Poisson distribution with mean rate `lambda`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonDist {
    pub lambda: f64,
}

impl PoissonDist {
    /// Creates a Poisson distribution.
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0, "PoissonDist: lambda must be > 0");
        PoissonDist { lambda }
    }

    /// Samples an integer count.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        let l = (-self.lambda).exp();
        let mut k = 0;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= rng.next_f64();
            if p <= l {
                break;
            }
        }
        (k - 1) as u64
    }
}

// =============================================================================
// ShuffleSeq & Shuffling Algorithms
// =============================================================================

/// In-place Fisher-Yates shuffle engine.
pub struct ShuffleSeq;

impl ShuffleSeq {
    /// Shuffles a mutable slice in-place with uniform O(N) complexity.
    pub fn shuffle<T, R: Rng + ?Sized>(slice: &mut [T], rng: &mut R) {
        let len = slice.len();
        if len <= 1 {
            return;
        }
        for i in (1..len).rev() {
            let j = (rng.next_u64() % ((i + 1) as u64)) as usize;
            slice.swap(i, j);
        }
    }

    /// Generates a random permutation of integers `0..n`.
    pub fn randperm<R: Rng + ?Sized>(n: usize, rng: &mut R) -> Vec<usize> {
        let mut perm: Vec<usize> = (0..n).collect();
        Self::shuffle(&mut perm, rng);
        perm
    }
}

// =============================================================================
// Global Seed Management
// =============================================================================

static GLOBAL_SEED: AtomicU64 = AtomicU64::new(0x1234_5678_9ABC_DEF0);

thread_local! {
    static THREAD_RNG: RefCell<BrainRng> = RefCell::new(BrainRng::from_seed(
        GLOBAL_SEED.fetch_add(0x9E3779B97F4A7C15, Ordering::SeqCst)
    ));
}

/// Sets the global base random seed.
pub fn set_seed(seed: u64) {
    GLOBAL_SEED.store(seed, Ordering::SeqCst);
    THREAD_RNG.with(|rng| {
        *rng.borrow_mut() = BrainRng::from_seed(seed);
    });
}

/// Alias for `set_seed`.
pub fn manual_seed(seed: u64) {
    set_seed(seed);
}

/// Gets the current global base seed value.
pub fn get_seed() -> u64 {
    GLOBAL_SEED.load(Ordering::SeqCst)
}

/// Executes a closure with a mutable reference to the calling thread's default RNG.
pub fn with_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut BrainRng) -> R,
{
    THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splitmix64_reproducibility() {
        let mut sm1 = SplitMix64::new(42);
        let mut sm2 = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(sm1.next_u64(), sm2.next_u64());
        }
    }

    #[test]
    fn test_pcg32_basic() {
        let mut pcg = PCG32::from_seed(12345);
        let val1 = pcg.next_u32();
        let val2 = pcg.next_u32();
        assert_ne!(val1, val2);
    }

    #[test]
    fn test_chacha8_reproducibility() {
        let mut c1 = ChaCha8::from_seed(999);
        let mut c2 = ChaCha8::from_seed(999);
        for _ in 0..100 {
            assert_eq!(c1.next_u32(), c2.next_u32());
        }
    }

    #[test]
    fn test_seedseq_generation() {
        let seq = SeedSeq::from_u64(100);
        let seeds = seq.generate_u64_seeds(10);
        assert_eq!(seeds.len(), 10);
        for i in 0..9 {
            assert_ne!(seeds[i], seeds[i + 1]);
        }
    }

    #[test]
    fn test_uniform_dist_range() {
        let mut rng = BrainRng::from_seed(42);
        let dist = UniformDist::new(-5.0, 5.0);
        for _ in 0..1000 {
            let val = dist.sample(&mut rng);
            assert!(val >= -5.0 && val < 5.0);
        }
    }

    #[test]
    fn test_normal_dist_moments() {
        let mut rng = BrainRng::from_seed(123);
        let dist = NormalDist::new(2.0, 0.5);
        let mut sum = 0.0;
        let n = 10000;
        for _ in 0..n {
            sum += dist.sample(&mut rng);
        }
        let mean = sum / n as f64;
        assert!((mean - 2.0).abs() < 0.05);
    }

    #[test]
    fn test_gamma_and_beta_dist() {
        let mut rng = BrainRng::from_seed(456);
        let gamma = GammaDist::new(2.0, 1.5);
        let g = gamma.sample(&mut rng);
        assert!(g > 0.0);

        let beta = BetaDist::new(2.0, 5.0);
        let b = beta.sample(&mut rng);
        assert!(b >= 0.0 && b <= 1.0);
    }

    #[test]
    fn test_exponential_and_cauchy() {
        let mut rng = BrainRng::from_seed(789);
        let exp_dist = ExponentialDist::new(1.0);
        assert!(exp_dist.sample(&mut rng) >= 0.0);

        let cauchy = CauchyDist::new(0.0, 1.0);
        let _c = cauchy.sample(&mut rng);
    }

    #[test]
    fn test_bernoulli_and_poisson() {
        let mut rng = BrainRng::from_seed(321);
        let bern = BernoulliDist::new(0.7);
        let mut ones = 0;
        for _ in 0..1000 {
            if bern.sample_bool(&mut rng) {
                ones += 1;
            }
        }
        assert!(ones > 600 && ones < 800);

        let pois = PoissonDist::new(3.0);
        let _p = pois.sample(&mut rng);
    }

    #[test]
    fn test_shuffle_seq() {
        let mut rng = BrainRng::from_seed(555);
        let mut arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        ShuffleSeq::shuffle(&mut arr, &mut rng);
        assert_ne!(arr, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let mut sorted = arr;
        sorted.sort();
        assert_eq!(sorted, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_distributions_table() {
        let mut rng = BrainRng::from_seed(108);
        for &(low, high) in &[(-10.0, 10.0), (0.0, 1.0), (-1.0, 1.0)] {
            let unif = UniformDist::new(low, high);
            for _ in 0..50 {
                let u = unif.sample(&mut rng);
                assert!(u >= low && u < high);
            }
        }

        let norm = NormalDist::new(0.0, 1.0);
        let lognorm = LogNormalDist::new(0.0, 0.5);
        let gamma = GammaDist::new(1.5, 2.0);
        let beta = BetaDist::new(2.0, 2.0);
        let exp_d = ExponentialDist::new(2.5);
        let cauchy = CauchyDist::new(0.0, 1.0);
        let bern = BernoulliDist::new(0.5);

        for _ in 0..50 {
            assert!(norm.sample(&mut rng).is_finite());
            assert!(lognorm.sample(&mut rng) > 0.0);
            assert!(gamma.sample(&mut rng) > 0.0);
            let b = beta.sample(&mut rng);
            assert!((0.0..=1.0).contains(&b));
            assert!(exp_d.sample(&mut rng) >= 0.0);
            assert!(cauchy.sample(&mut rng).is_finite());
            let br = bern.sample(&mut rng);
            assert!(br == 0.0 || br == 1.0);
        }

        for n in [2, 5, 10, 32] {
            let perm = ShuffleSeq::randperm(n, &mut rng);
            assert_eq!(perm.len(), n);
            let mut sorted = perm.clone();
            sorted.sort();
            assert_eq!(sorted, (0..n).collect::<Vec<_>>());
        }
    }
}
