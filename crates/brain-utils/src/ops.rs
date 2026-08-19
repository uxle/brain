//! # Operational Utilities & Execution Modifiers
//!
//! Provides execution measuring wrappers, exponential backoff retries,
//! rate limiters, batching helpers, and closure adapters.

use std::thread;
use std::time::{Duration, Instant};
use crate::core::{UtilsError, UtilsResult};

/// Measures the execution duration of a closure and returns (result, duration).
pub fn measure_block<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let res = f();
    (res, start.elapsed())
}

/// Executes an operation with exponential backoff retry.
pub fn retry_with_backoff<F, T>(
    mut op: F,
    max_attempts: usize,
    initial_delay: Duration,
    max_delay: Duration,
    backoff_factor: f64,
) -> UtilsResult<T>
where
    F: FnMut(usize) -> UtilsResult<T>,
{
    if max_attempts == 0 {
        return Err(UtilsError::Unsupported("max_attempts must be > 0".to_string()));
    }
    let mut current_delay = initial_delay;
    for attempt in 1..=max_attempts {
        match op(attempt) {
            Ok(val) => return Ok(val),
            Err(e) => {
                if attempt == max_attempts {
                    return Err(e);
                }
                thread::sleep(current_delay);
                let next_millis = (current_delay.as_millis() as f64 * backoff_factor) as u64;
                current_delay = Duration::from_millis(next_millis).min(max_delay);
            }
        }
    }
    Err(UtilsError::Timeout("Retry loop exhausted".to_string()))
}

/// Token bucket rate limiter for controlling task throughput.
#[derive(Debug, Clone)]
pub struct TokenBucketRateLimiter {
    capacity: f64,
    tokens: f64,
    refill_rate_per_sec: f64,
    last_refill: Instant,
}

impl TokenBucketRateLimiter {
    /// Constructs a rate limiter with maximum burst capacity and refill rate per second.
    pub fn new(capacity: f64, refill_rate_per_sec: f64) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_rate_per_sec,
            last_refill: Instant::now(),
        }
    }

    /// Attempts to acquire n tokens. Returns true if acquired, false otherwise.
    pub fn try_acquire(&mut self, n: f64) -> bool {
        self.refill();
        if self.tokens >= n {
            self.tokens -= n;
            true
        } else {
            false
        }
    }

    /// Refills tokens based on elapsed time.
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate_per_sec).min(self.capacity);
        self.last_refill = now;
    }

    /// Current available token count.
    pub fn available_tokens(&mut self) -> f64 {
        self.refill();
        self.tokens
    }
}

/// Splits a slice of items into fixed-size chunk batches.
pub fn chunk_slice<T: Clone>(items: &[T], chunk_size: usize) -> Vec<Vec<T>> {
    if chunk_size == 0 || items.is_empty() {
        return Vec::new();
    }
    items.chunks(chunk_size).map(|c| c.to_vec()).collect()
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_ops_retry_and_limiter_1() {
        let (res, dur) = measure_block(|| {
            let mut sum = 0u64;
            for k in 0..1000 { sum += k; }
            sum
        });
        assert_eq!(res, 499500);
        let _ = dur;
    
        let mut attempts_done = 0;
        let retry_res = retry_with_backoff(
            |attempt| {
                attempts_done = attempt;
                if attempt < 3 {
                    Err(UtilsError::IoError("temp fail".into()))
                } else {
                    Ok("success")
                }
            },
            5,
            Duration::from_millis(1),
            Duration::from_millis(10),
            1.5,
        );
        assert_eq!(retry_res.unwrap(), "success");
        assert_eq!(attempts_done, 3);
    
        let mut limiter = TokenBucketRateLimiter::new(10.0, 5.0);
        assert!(limiter.try_acquire(5.0));
        assert!(limiter.try_acquire(5.0));
        assert!(!limiter.try_acquire(1.0));
    
        let chunks = chunk_slice(&[1, 2, 3, 4, 5, 6, 7], 3);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], vec![1, 2, 3]);
        assert_eq!(chunks[2], vec![7]);
    }
}
