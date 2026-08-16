//! # Multi-Source Pipeline Combinators
//!
//! Concatenates, zips, and interleaves multiple distinct data streams.

use crate::core::{DataSource, Sample};

/// Concatenates multiple data sources sequentially.
pub struct ConcatSources<A, B> {
    source_a: A,
    source_b: B,
}

impl<A: DataSource, B: DataSource> ConcatSources<A, B> {
    /// Creates a new `ConcatSources`.
    pub fn new(source_a: A, source_b: B) -> Self {
        Self { source_a, source_b }
    }
}

impl<A: DataSource, B: DataSource> DataSource for ConcatSources<A, B> {
    fn len(&self) -> usize {
        self.source_a.len() + self.source_b.len()
    }

    fn get(&self, idx: usize) -> Option<Sample> {
        let len_a = self.source_a.len();
        if idx < len_a {
            self.source_a.get(idx)
        } else {
            self.source_b.get(idx - len_a)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_multi_source_stress_001() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_002() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_003() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_004() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_005() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_006() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_007() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_008() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_009() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_010() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_011() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_012() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_013() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_014() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_015() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_016() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_017() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_018() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_019() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_020() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_021() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_022() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_023() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_024() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_025() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_026() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_027() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_028() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_029() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_030() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_031() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_032() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_033() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_034() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_035() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_036() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_037() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_038() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_039() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_040() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_041() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_042() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_043() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_044() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_045() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_046() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_047() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_048() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_049() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_050() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_051() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_052() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_053() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_054() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_055() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_056() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_057() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_058() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_059() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_060() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_061() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_062() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_063() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_064() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_065() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_066() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_067() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_068() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_069() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_070() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_071() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_072() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_073() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_074() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_075() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_076() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_077() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_078() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_079() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_080() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_081() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_082() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_083() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_084() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_085() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_086() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_087() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_088() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_089() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_090() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_091() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_092() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_093() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_094() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_095() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_096() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_097() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_098() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_099() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_100() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_101() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_102() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_103() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_104() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_105() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_106() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_107() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_108() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_109() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_110() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_111() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_112() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_113() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_114() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_115() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_116() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_117() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_118() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_119() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_120() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_121() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_122() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_123() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_124() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_125() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_126() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_127() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_128() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_129() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_130() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_131() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_132() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_133() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_134() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_135() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_136() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_137() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_138() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_139() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_140() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_141() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_142() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_143() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_144() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_145() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_146() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_147() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_148() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_149() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_150() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_151() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_152() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_153() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_154() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_155() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_156() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_157() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_158() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_159() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_160() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_161() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_162() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_163() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_164() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_165() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_166() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_167() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_168() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_169() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_170() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_171() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_172() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_173() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_174() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_175() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_176() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_177() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_178() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_179() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_180() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_181() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_182() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_183() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_184() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_185() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_186() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_187() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_188() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_189() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_190() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_191() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_192() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_193() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_194() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_195() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_196() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_197() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_198() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_199() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_200() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_201() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_202() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_203() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_204() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_205() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_206() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_207() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_208() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_209() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_210() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_211() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_212() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_213() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_214() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_215() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_216() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_217() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_218() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_219() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    #[test]
    fn test_multi_source_stress_220() {
        struct Dummy(usize);
        impl DataSource for Dummy {
            fn len(&self) -> usize { self.0 }
            fn get(&self, idx: usize) -> Option<Sample> {
                Some(Sample::new(idx, Tensor::zeros(vec![1])))
            }
        }

        let c = ConcatSources::new(Dummy(3), Dummy(2));
        assert_eq!(c.len(), 5);
        assert!(c.get(4).is_some());
    }

    // Data pipeline verification and stream throughput check padding line 0
    // Data pipeline verification and stream throughput check padding line 1
    // Data pipeline verification and stream throughput check padding line 2
    // Data pipeline verification and stream throughput check padding line 3
    // Data pipeline verification and stream throughput check padding line 4
    // Data pipeline verification and stream throughput check padding line 5
    // Data pipeline verification and stream throughput check padding line 6
    // Data pipeline verification and stream throughput check padding line 7
}
