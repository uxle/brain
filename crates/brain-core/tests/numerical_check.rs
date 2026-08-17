//! Numerical correctness tests for core brain-core tensor ops.

use brain_core::Tensor;
use brain_core::tensor::arithmetic as arith;
use brain_core::tensor::conv;
use brain_core::tensor::pool;
use brain_core::tensor::reduction as red;
use brain_core::tensor::special as spec;

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn check_transpose_roundtrip() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let t = Tensor::from_slice(&data, vec![2, 3]);
    let tt = t.transpose(0, 1);
    // tt is 3x2: [[1,4],[2,5],[3,6]]
    assert_eq!(tt.shape(), &[3, 2]);
    assert!(approx(tt.get_2d(0, 0), 1.0));
    assert!(approx(tt.get_2d(0, 1), 4.0));
    assert!(approx(tt.get_2d(2, 1), 6.0));
    // transpose transpose back
    let ttt = tt.transpose(0, 1);
    assert_eq!(ttt.shape(), &[2, 3]);
    assert!(approx(ttt.get_2d(0, 0), 1.0));
    assert!(approx(ttt.get_2d(1, 2), 6.0));
    println!("OK [transpose] roundtrip correct");
}

#[test]
fn check_reshape() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let r = t.reshape(vec![3, 2]);
    assert_eq!(r.shape(), &[3, 2]);
    // data should be preserved in row-major
    assert_eq!(r.to_vec(), t.to_vec());
    let r2 = r.reshape(vec![6]);
    assert_eq!(r2.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("OK [reshape] preserves data");
}

#[test]
fn check_permute() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
    let p = t.permute(&[2, 0, 1]); // [3, 1, 2]
    assert_eq!(p.shape(), &[3, 1, 2]);
    println!("OK [permute] shape {:?}", p.shape());
}

#[test]
fn check_reduce_along_dim_sum() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let s = red::sum_along_dim(&t, 1, false);
    // sum along rows: [6, 15]
    assert_eq!(s.shape(), &[2]);
    assert!(approx(s.get(0), 6.0));
    assert!(approx(s.get(1), 15.0));
    let s0 = red::sum_along_dim(&t, 0, false);
    println!("sum_along_dim dim0 = {:?}", s0.to_vec());
    // sum along cols: [1+4, 2+5, 3+6] = [5,7,9]
    assert_eq!(s0.shape(), &[3]);
    assert!(approx(s0.get(0), 5.0));
    assert!(approx(s0.get(1), 7.0));
    assert!(approx(s0.get(2), 9.0));
    println!("OK [sum_along_dim] rows and cols");
}

#[test]
fn check_reduce_along_dim_max() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let m = red::max_along_dim(&t, 1, false);
    assert_eq!(m.shape(), &[2]);
    assert!(approx(m.get(0), 3.0));
    assert!(approx(m.get(1), 6.0));
    println!("OK [max_along_dim]");
}

#[test]
fn check_softmax_correctness() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3]);
    let sm = spec::softmax(&t, 1);
    let s = red::sum(&sm);
    assert!(approx(s, 1.0));
    // softmax([1,2,3]): e/(e+e^2+e^3) etc
    let e1 = 2.718281828459045_f64;
    let e2 = e1 * e1;
    let e3 = e2 * e1;
    let z = e1 + e2 + e3;
    assert!(approx(sm.get(0), e1 / z));
    assert!(approx(sm.get(1), e2 / z));
    assert!(approx(sm.get(2), e3 / z));
    println!("OK [softmax] values correct");
}

#[test]
fn check_max_pool2d() {
    let input = Tensor::from_slice(
        &[1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
        vec![1, 1, 4, 2],
    );
    let out = pool::max_pool2d(&input, (2, 2), (2, 2), (0, 0));
    assert_eq!(out.shape(), &[1, 1, 2, 1]);
    // input 4x2:
    // [[1,3],[2,4],[5,7],[6,8]]
    // max pool 2x2 stride 2: top-left window [1,3,2,4] -> 4, bottom [5,7,6,8] -> 8
    assert!(approx(out.get(0), 4.0));
    assert!(approx(out.get(1), 8.0));
    println!("OK [max_pool2d]");
}

#[test]
fn check_avg_pool2d_no_pad() {
    let input = Tensor::from_slice(
        &[1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
        vec![1, 1, 4, 2],
    );
    let out = pool::avg_pool2d(&input, (2, 2), (2, 2), (0, 0));
    assert_eq!(out.shape(), &[1, 1, 2, 1]);
    // top window: (1+3+2+4)/4 = 2.5
    // bottom: (5+7+6+8)/4 = 6.5
    assert!(approx(out.get(0), 2.5));
    assert!(approx(out.get(1), 6.5));
    println!("OK [avg_pool2d] no padding");
}

#[test]
fn check_avg_pool2d_with_pad() {
    // input 3x3, kernel 2x2, stride 1, pad 1 => out 4x4
    let input = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
    );
    let out = pool::avg_pool2d(&input, (2, 2), (1, 1), (1, 1));
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
    // Check a corner window: top-left, with padding zeros, the 2x2 window covers
    // padded (0,0), (0,1=input[0]), (1,0=input[0]), (1,1=input[1])? Actually pad=1 zero.
    // With count_include_pad semantics the divisor differs. Let's just print to inspect.
    println!("avg_pool2d pad corner out[0,0,0,0] = {}", out.get(0));
    println!("avg_pool2d pad out = {:?}", out.to_vec());
}

#[test]
fn check_global_avg_pool2d() {
    let input = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![1, 1, 2, 4],
    );
    let out = pool::global_avg_pool2d(&input);
    assert_eq!(out.shape(), &[1, 1, 1, 1]);
    assert!(approx(out.get(0), 4.5));
    println!("OK [global_avg_pool2d] = {}", out.get(0));
}

#[test]
fn check_batched_matmul_broadcast() {
    // a: [1, 2, 3], b: [2, 3, 4] -> broadcast batch dim -> [2, 2, 4]
    let a_data: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let a = Tensor::from_slice(&a_data, vec![1, 2, 3]);
    let b_data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let b = Tensor::from_slice(&b_data, vec![2, 3, 4]);
    let c = arith::matmul(&a, &b);
    println!("batched matmul broadcast shape = {:?}", c.shape());
    assert_eq!(c.shape(), &[2, 2, 4]);

    // Reference: a matrix [[0,1,2],[3,4,5]] broadcast to both b's batch matrices.
    let a_mat = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    for bb in 0..2 {
        let b_mat = if bb == 0 {
            (0..12).map(|i| i as f64).collect::<Vec<_>>()
        } else {
            (12..24).map(|i| i as f64).collect::<Vec<_>>()
        };
        for i in 0..2 {
            for j in 0..4 {
                let mut expected = 0.0_f64;
                for k in 0..3 {
                    expected += a_mat[i * 3 + k] * b_mat[k * 4 + j];
                }
                let got = c.get_3d(bb, i, j);
                assert!(approx(got, expected),
                    "matmul broadcast mismatch bb={bb} i={i} j={j}: got {got} expected {expected}");
            }
        }
    }
    println!("OK [batched_matmul_broadcast] values verified");
}

#[test]
fn check_arange() {
    let asc = Tensor::arange(0.0, 10.0, 2.0);
    assert_eq!(asc.to_vec(), vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    let desc = Tensor::arange(0.0, -5.0, -1.0);
    assert_eq!(desc.to_vec(), vec![0.0, -1.0, -2.0, -3.0, -4.0]);
    println!("OK [arange] asc and desc");
}

#[test]
fn check_conv2d_output_size() {
    // standard case: input 5x5, kernel 3x3, stride 1, pad 0 => out 3x3
    let input = Tensor::zeros(vec![1, 1, 5, 5]);
    let weight = Tensor::ones(vec![1, 1, 3, 3]);
    let out = conv::conv2d(&input, &weight, None, (1, 1), (0, 0));
    assert_eq!(out.shape(), &[1, 1, 3, 3]);
    println!("OK [conv2d] 5x5->3x3");

    // with padding: input 5x5, kernel 3x3, stride 2, pad 1 => out 3x3
    let out2 = conv::conv2d(&input, &weight, None, (2, 2), (1, 1));
    assert_eq!(out2.shape(), &[1, 1, 3, 3]);
    println!("OK [conv2d] pad+stride out {:?}", out2.shape());
}

#[test]
fn check_avg_pool_counts_valid_only() {
    // When padding adds zero elements to a window, the average should ideally
    // divide by the number of valid (non-padded) elements. Check behavior.
    let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![1, 1, 3, 3]);
    let out = pool::avg_pool2d(&input, (2, 2), (1, 1), (1, 1));
    println!("avg_pool2d 3x3 with pad=1 2x2 stride1 out = {:?}", out.to_vec());
}

#[test]
fn check_dtype_is_lossless_cast() {
    use brain_core::dtype::DType;
    // U64 + I32: different size, U64 bigger => U64 (both same sign category? no, unsigned vs signed, different size)
    let p = DType::promote(DType::U64, DType::I32);
    println!("promote(U64, I32) = {:?}", p);

    // Lossless cast checks
    // F16 -> BF16: BF16 has fewer mantissa bits (7) than F16 (10), so NOT lossless
    let f16_to_bf16 = DType::BF16.is_lossless_cast(DType::F16);
    println!("is_lossless_cast(BF16 <- F16) = {} (expected false: BF16 has less mantissa precision)", f16_to_bf16);

    // I8 -> I16: lossless
    let i8_to_i16 = DType::I16.is_lossless_cast(DType::I8);
    println!("is_lossless_cast(I16 <- I8) = {} (expected true)", i8_to_i16);

    // I16 -> I8: not lossless
    let i16_to_i8 = DType::I8.is_lossless_cast(DType::I16);
    println!("is_lossless_cast(I8 <- I16) = {} (expected false)", i16_to_i8);

    // U8 -> I16: lossless (U8 in [0,255] fits in I16)
    let u8_to_i16 = DType::I16.is_lossless_cast(DType::U8);
    println!("is_lossless_cast(I16 <- U8) = {} (expected true)", u8_to_i16);

    // I32 -> F32: NOT lossless (i32 up to 2^31 needs 32 bits, f32 has 24 mantissa bits)
    let i32_to_f32 = DType::F32.is_lossless_cast(DType::I32);
    println!("is_lossless_cast(F32 <- I32) = {} (expected false: I32 can exceed f32 precision)", i32_to_f32);
}
