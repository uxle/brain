//! # Declarative Benchmark Macros
//!
//! Provides proc-macro-free declarative macros (`bench_case!`, `bench_group!`) for defining benchmark cases.

/// Defines an inline benchmark function.
#[macro_export]
macro_rules! bench_case {
    ($name:ident, $body:expr) => {
        pub fn $name() -> $crate::core::BenchResult {
            let config = $crate::core::BenchConfig::new(stringify!($name));
            $crate::runner::Runner::run_benchmark(&config, || $body).unwrap()
        }
    };
}

/// Groups several benchmark definitions together.
#[macro_export]
macro_rules! bench_group {
    ($group_name:ident, $($name:ident => $body:expr),* $(,)?) => {
        pub fn $group_name() -> Vec<$crate::core::BenchResult> {
            let mut results = Vec::new();
            $(
                let config = $crate::core::BenchConfig::new(stringify!($name));
                results.push($crate::runner::Runner::run_benchmark(&config, || $body).unwrap());
            )*
            results
        }
    };
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
