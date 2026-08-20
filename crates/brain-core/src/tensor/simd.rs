//! SIMD vectorization abstractions and unrolled numerical kernels.
//!
//! Features dynamic AVX2/FMA hardware acceleration on x86_64 with clean portable fallbacks.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// =============================================================================
// AVX2 / FMA Vector Kernels (x86_64)
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn simd_fma_avx2(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
    unsafe {
        let len = a.len().min(b.len()).min(c.len()).min(out.len());
        let mut i = 0;
        while i + 4 <= len {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            let vc = _mm256_loadu_pd(c.as_ptr().add(i));
            let vres = _mm256_fmadd_pd(va, vb, vc);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), vres);
            i += 4;
        }
        while i < len {
            *out.get_unchecked_mut(i) =
                *a.get_unchecked(i) * *b.get_unchecked(i) + *c.get_unchecked(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn simd_dot_avx2(a: &[f64], b: &[f64]) -> f64 {
    unsafe {
        let len = a.len().min(b.len());
        let mut vsum = _mm256_setzero_pd();
        let mut i = 0;
        while i + 4 <= len {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            vsum = _mm256_fmadd_pd(va, vb, vsum);
            i += 4;
        }
        let mut buf = [0.0f64; 4];
        _mm256_storeu_pd(buf.as_mut_ptr(), vsum);
        let mut total = buf[0] + buf[1] + buf[2] + buf[3];
        while i < len {
            total += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        total
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_add_avx2(a: &[f64], b: &[f64], out: &mut [f64]) {
    unsafe {
        let len = a.len().min(b.len()).min(out.len());
        let mut i = 0;
        while i + 4 <= len {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            let vres = _mm256_add_pd(va, vb);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), vres);
            i += 4;
        }
        while i < len {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_mul_avx2(a: &[f64], b: &[f64], out: &mut [f64]) {
    unsafe {
        let len = a.len().min(b.len()).min(out.len());
        let mut i = 0;
        while i + 4 <= len {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            let vres = _mm256_mul_pd(va, vb);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), vres);
            i += 4;
        }
        while i < len {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
    }
}

// =============================================================================
// Public Dispatch Functions (Dynamic Runtime CPU Detection)
// =============================================================================

/// Vector addition: out[i] = a[i] + b[i]
pub fn simd_add(a: &[f64], b: &[f64], out: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            simd_add_avx2(a, b, out);
            return;
        }
    }

    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;
    while i + 4 <= len {
        out[i] = a[i] + b[i];
        out[i + 1] = a[i + 1] + b[i + 1];
        out[i + 2] = a[i + 2] + b[i + 2];
        out[i + 3] = a[i + 3] + b[i + 3];
        i += 4;
    }
    while i < len {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

/// Vector subtraction: out[i] = a[i] - b[i]
pub fn simd_sub(a: &[f64], b: &[f64], out: &mut [f64]) {
    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;
    while i + 4 <= len {
        out[i] = a[i] - b[i];
        out[i + 1] = a[i + 1] - b[i + 1];
        out[i + 2] = a[i + 2] - b[i + 2];
        out[i + 3] = a[i + 3] - b[i + 3];
        i += 4;
    }
    while i < len {
        out[i] = a[i] - b[i];
        i += 1;
    }
}

/// Vector multiplication: out[i] = a[i] * b[i]
pub fn simd_mul(a: &[f64], b: &[f64], out: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            simd_mul_avx2(a, b, out);
            return;
        }
    }

    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;
    while i + 4 <= len {
        out[i] = a[i] * b[i];
        out[i + 1] = a[i + 1] * b[i + 1];
        out[i + 2] = a[i + 2] * b[i + 2];
        out[i + 3] = a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < len {
        out[i] = a[i] * b[i];
        i += 1;
    }
}

/// Vector fused multiply-add: out[i] = a[i] * b[i] + c[i]
pub fn simd_fma(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe {
            simd_fma_avx2(a, b, c, out);
            return;
        }
    }

    let len = a.len().min(b.len()).min(c.len()).min(out.len());
    let mut i = 0;
    while i + 4 <= len {
        out[i] = a[i] * b[i] + c[i];
        out[i + 1] = a[i + 1] * b[i + 1] + c[i + 1];
        out[i + 2] = a[i + 2] * b[i + 2] + c[i + 2];
        out[i + 3] = a[i + 3] * b[i + 3] + c[i + 3];
        i += 4;
    }
    while i < len {
        out[i] = a[i] * b[i] + c[i];
        i += 1;
    }
}

/// Vector dot product
pub fn simd_dot(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe {
            return simd_dot_avx2(a, b);
        }
    }

    let len = a.len().min(b.len());
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let mut i = 0;
    while i + 4 <= len {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut total = sum0 + sum1 + sum2 + sum3;
    while i < len {
        total += a[i] * b[i];
        i += 1;
    }
    total
}

/// Vector ReLU: out[i] = max(0.0, a[i])
pub fn simd_relu(a: &[f64], out: &mut [f64]) {
    let len = a.len().min(out.len());
    let mut i = 0;
    while i + 4 <= len {
        out[i] = a[i].max(0.0);
        out[i + 1] = a[i + 1].max(0.0);
        out[i + 2] = a[i + 2].max(0.0);
        out[i + 3] = a[i + 3].max(0.0);
        i += 4;
    }
    while i < len {
        out[i] = a[i].max(0.0);
        i += 1;
    }
}

/// Vector Scale and Add: out[i] += alpha * a[i]
pub fn simd_axpy(alpha: f64, a: &[f64], out: &mut [f64]) {
    let len = a.len().min(out.len());
    let mut i = 0;
    while i + 4 <= len {
        out[i] += alpha * a[i];
        out[i + 1] += alpha * a[i + 1];
        out[i + 2] += alpha * a[i + 2];
        out[i + 3] += alpha * a[i + 3];
        i += 4;
    }
    while i < len {
        out[i] += alpha * a[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_add_and_sub() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = [0.0; 5];
        simd_add(&a, &b, &mut out);
        assert_eq!(out, [11.0, 22.0, 33.0, 44.0, 55.0]);

        simd_sub(&b, &a, &mut out);
        assert_eq!(out, [9.0, 18.0, 27.0, 36.0, 45.0]);
    }

    #[test]
    fn test_simd_fma_and_dot() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 2.0, 2.0, 2.0, 2.0];
        let c = [10.0, 10.0, 10.0, 10.0, 10.0];
        let mut out = [0.0; 5];
        simd_fma(&a, &b, &c, &mut out);
        assert_eq!(out, [12.0, 14.0, 16.0, 18.0, 20.0]);

        let d = simd_dot(&a, &b);
        assert_eq!(d, 30.0);
    }
}
