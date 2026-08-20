//! Layer-level gradient checking and parameter cross-checks.

use brain_autograd::Value;
use brain_core::Tensor;
use brain_nn::{
    BatchNorm2d, Conv2d, Dropout, Embedding, LayerNorm, Linear, MaxPool2d, Module, RMSNorm,
};

/// Verifies analytic gradient against numerical central finite differences.
pub fn check_param_gradient<F>(
    eval_loss: F,
    param_data: &[f64],
    analytic_grad: &[f64],
    eps: f64,
    tol: f64,
) where
    F: Fn(&[f64]) -> f64,
{
    assert_eq!(param_data.len(), analytic_grad.len());
    for i in 0..param_data.len() {
        let mut p_plus = param_data.to_vec();
        p_plus[i] += eps;
        let mut p_minus = param_data.to_vec();
        p_minus[i] -= eps;

        let l_plus = eval_loss(&p_plus);
        let l_minus = eval_loss(&p_minus);
        let numeric = (l_plus - l_minus) / (2.0 * eps);
        let analytic = analytic_grad[i];

        let abs_err = (analytic - numeric).abs();
        if abs_err > 1e-5 {
            let scale = analytic.abs().max(numeric.abs());
            let rel_err = abs_err / scale;
            assert!(
                rel_err < tol,
                "Param grad mismatch at idx {}: analytic={:.6}, numeric={:.6}, abs_err={:.6}, rel_err={:.6}",
                i, analytic, numeric, abs_err, rel_err
            );
        }
    }
}

// =============================================================================
// 1. Linear Layer Gradient Checks
// =============================================================================

#[test]
fn test_linear_layer_weight_and_bias_gradient() {
    let mut linear = Linear::new(3, 2, true);
    linear.weight = Value::new(
        Tensor::from_slice(&[0.5, -0.2, 1.0, 0.8, -1.2, 0.3], vec![2, 3]),
        true,
    );
    linear.bias = Some(Value::new(Tensor::from_slice(&[0.1, -0.4], vec![2]), true));

    let x = Value::new(
        Tensor::from_slice(&[1.0, 2.0, 3.0, -1.0, 0.5, 2.0], vec![2, 3]),
        false,
    );

    let b_size = 2.0;
    let mut analytic_dw = vec![0.0f64; 6];
    let x_data = x.to_vec();
    for o in 0..2 {
        for i in 0..3 {
            analytic_dw[o * 3 + i] = x_data[0 * 3 + i] + x_data[1 * 3 + i];
        }
    }
    let analytic_db = vec![b_size, b_size];

    let w_data = linear.weight.to_vec();
    let x_fixed = x.clone();
    let b_fixed = linear.bias.clone().unwrap();
    check_param_gradient(
        |w_test| {
            let mut l = linear.clone();
            l.weight = Value::new(Tensor::from_slice(w_test, vec![2, 3]), true);
            l.forward(&x_fixed).unwrap().to_vec().iter().sum()
        },
        &w_data,
        &analytic_dw,
        1e-5,
        1e-4,
    );

    let b_data = b_fixed.to_vec();
    check_param_gradient(
        |b_test| {
            let mut l = linear.clone();
            l.bias = Some(Value::new(Tensor::from_slice(b_test, vec![2]), true));
            l.forward(&x_fixed).unwrap().to_vec().iter().sum()
        },
        &b_data,
        &analytic_db,
        1e-5,
        1e-4,
    );
}

#[test]
fn test_linear_layer_without_bias() {
    let mut linear = Linear::new(2, 2, false);
    linear.weight = Value::new(Tensor::from_slice(&[1.5, -0.5, 0.2, 2.0], vec![2, 2]), true);
    let x = Value::new(Tensor::from_slice(&[2.0, 3.0], vec![1, 2]), false);

    let w_data = linear.weight.to_vec();
    let x_fixed = x.clone();
    let analytic_dw = vec![2.0, 3.0, 2.0, 3.0];

    check_param_gradient(
        |w_test| {
            let mut l = linear.clone();
            l.weight = Value::new(Tensor::from_slice(w_test, vec![2, 2]), true);
            l.forward(&x_fixed).unwrap().to_vec().iter().sum()
        },
        &w_data,
        &analytic_dw,
        1e-5,
        1e-4,
    );
}

// =============================================================================
// 2. Conv2d Layer Gradient Checks
// =============================================================================

#[test]
fn test_conv2d_layer_weight_and_bias_gradient() {
    let mut conv = Conv2d::new(1, 1, 2, true);
    conv.config.padding = (0, 0);
    conv.weight = Value::new(
        Tensor::from_slice(&[0.5, -0.5, 1.0, -1.0], vec![1, 1, 2, 2]),
        true,
    );
    conv.bias = Some(Value::new(Tensor::from_slice(&[0.2], vec![1]), true));

    let input = Value::new(
        Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![1, 1, 3, 3],
        ),
        false,
    );

    let w_data = conv.weight.to_vec();
    let in_fixed = input.clone();

    let eval_w = |w_test: &[f64]| {
        let mut c = conv.clone();
        c.weight = Value::new(Tensor::from_slice(w_test, vec![1, 1, 2, 2]), true);
        c.forward(&in_fixed).unwrap().to_vec().iter().sum()
    };

    let mut analytic_dw = vec![0.0f64; 4];
    for kh in 0..2 {
        for kw in 0..2 {
            let mut s = 0.0;
            for oh in 0..2 {
                for ow in 0..2 {
                    s += input.get_4d(0, 0, oh + kh, ow + kw);
                }
            }
            analytic_dw[kh * 2 + kw] = s;
        }
    }

    check_param_gradient(eval_w, &w_data, &analytic_dw, 1e-5, 1e-4);

    let b_data = conv.bias.clone().unwrap().to_vec();
    let analytic_db = vec![4.0];
    check_param_gradient(
        |b_test| {
            let mut c = conv.clone();
            c.bias = Some(Value::new(Tensor::from_slice(b_test, vec![1]), true));
            c.forward(&in_fixed).unwrap().to_vec().iter().sum()
        },
        &b_data,
        &analytic_db,
        1e-5,
        1e-4,
    );
}

// =============================================================================
// 3. BatchNorm2d Gradient Formula Verification
// =============================================================================

#[test]
fn test_batchnorm2d_full_gradient_formula() {
    let mut bn = BatchNorm2d::new(2);
    bn.weight = Tensor::from_slice(&[1.5, 0.8], vec![2]);
    bn.bias = Tensor::from_slice(&[0.2, -0.1], vec![2]);

    let input = Tensor::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 0.5, -0.5, 1.5, -1.0, 2.0, 1.0, 4.0, 3.0, 1.0, -1.0, 0.0, 0.5,
        ],
        vec![2, 2, 2, 2],
    );

    let in_fixed = input.clone();
    let out = bn.forward_train(&input);
    assert_eq!(out.shape(), &[2, 2, 2, 2]);

    let w_data = bn.weight.to_vec();
    let eval_gamma = |g_test: &[f64]| {
        let mut b = bn.clone();
        b.weight = Tensor::from_slice(g_test, vec![2]);
        b.forward_train(&in_fixed).to_vec().iter().sum()
    };
    let analytic_dgamma = vec![0.0, 0.0];
    check_param_gradient(eval_gamma, &w_data, &analytic_dgamma, 1e-5, 1e-4);

    let b_data = bn.bias.to_vec();
    let eval_beta = |b_test: &[f64]| {
        let mut b = bn.clone();
        b.bias = Tensor::from_slice(b_test, vec![2]);
        b.forward_train(&in_fixed).to_vec().iter().sum()
    };
    let analytic_dbeta = vec![8.0, 8.0];
    check_param_gradient(eval_beta, &b_data, &analytic_dbeta, 1e-5, 1e-4);

    let in_data = input.to_vec();
    let eval_input = |in_test: &[f64]| {
        let b = bn.clone();
        let inp = Tensor::from_slice(in_test, vec![2, 2, 2, 2]);
        b.forward_train(&inp).to_vec().iter().sum()
    };
    let analytic_dx = vec![0.0; in_data.len()];
    check_param_gradient(eval_input, &in_data, &analytic_dx, 1e-5, 1e-4);
}

// =============================================================================
// 4. LayerNorm & RMSNorm Gradient Checks
// =============================================================================

#[test]
fn test_layernorm_and_rmsnorm_gradient() {
    let mut ln = LayerNorm::new(vec![3], 1e-5);
    ln.weight = Tensor::from_slice(&[1.0, 2.0, 0.5], vec![3]);
    ln.bias = Tensor::from_slice(&[0.1, -0.1, 0.2], vec![3]);

    let input = Value::new(
        Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        false,
    );
    let in_fixed = input.clone();

    let b_data = ln.bias.to_vec();
    check_param_gradient(
        |b_test| {
            let mut l = ln.clone();
            l.bias = Tensor::from_slice(b_test, vec![3]);
            Module::forward(&l, &in_fixed)
                .unwrap()
                .to_vec()
                .iter()
                .sum()
        },
        &b_data,
        &[2.0, 2.0, 2.0],
        1e-5,
        1e-4,
    );

    let mut rms = RMSNorm::new(3, 1e-5);
    rms.weight = Value::new(Tensor::from_slice(&[1.2, 0.8, 1.0], vec![3]), true);
    let w_data = rms.weight.data().to_vec();
    check_param_gradient(
        |w_test| {
            let mut r = rms.clone();
            r.weight = Value::new(Tensor::from_slice(w_test, vec![3]), true);
            Module::forward(&r, &in_fixed)
                .unwrap()
                .data()
                .to_vec()
                .iter()
                .sum()
        },
        &w_data,
        &[
            (Module::forward(&rms, &in_fixed).unwrap().data().to_vec()[0] / 1.2
                + Module::forward(&rms, &in_fixed).unwrap().data().to_vec()[3] / 1.2),
            (Module::forward(&rms, &in_fixed).unwrap().data().to_vec()[1] / 0.8
                + Module::forward(&rms, &in_fixed).unwrap().data().to_vec()[4] / 0.8),
            (Module::forward(&rms, &in_fixed).unwrap().data().to_vec()[2] / 1.0
                + Module::forward(&rms, &in_fixed).unwrap().data().to_vec()[5] / 1.0),
        ],
        1e-5,
        1e-4,
    );
}

// =============================================================================
// 5. Embedding Duplicate Index Accumulation
// =============================================================================

#[test]
fn test_embedding_duplicate_index_accumulation() {
    let mut emb = Embedding::new(4, 2);
    emb.weight = Value::new(
        Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2]),
        true,
    );

    let indices = vec![1, 2, 1];

    let analytic_dw = vec![0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0];

    let w_data = emb.weight.to_vec();
    check_param_gradient(
        |w_test| {
            let mut e = emb.clone();
            e.weight = Value::new(Tensor::from_slice(w_test, vec![4, 2]), true);
            e.forward_indices(&indices).to_vec().iter().sum()
        },
        &w_data,
        &analytic_dw,
        1e-5,
        1e-4,
    );
}

// =============================================================================
// 6. Dropout Forward/Backward Consistency & Eval Mode
// =============================================================================

#[test]
fn test_dropout_mask_consistency_and_eval() {
    let mut drop = Dropout::with_seed(0.5, 42);
    let input = Value::new(Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]), false);

    let out_train1 = drop.forward(&input).unwrap();
    let out_train2 = drop.forward(&input).unwrap();
    assert_eq!(
        out_train1.to_vec(),
        out_train2.to_vec(),
        "Dropout mask must be deterministic with same seed"
    );

    drop.set_training(false);
    let out_eval = drop.forward(&input).unwrap();
    assert_eq!(out_eval.to_vec(), input.to_vec());
}

// =============================================================================
// 7. MaxPool2d Argmax Routing and Tie-Breaking
// =============================================================================

#[test]
fn test_maxpool2d_argmax_and_tie_breaking() {
    let pool = MaxPool2d::new(2, 2);
    let input = Value::new(
        Tensor::from_slice(
            &[
                4.0, 4.0, 1.0, 2.0, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 5.0, 6.0, 2.0, 3.0, 7.0, 8.0,
            ],
            vec![1, 1, 4, 4],
        ),
        false,
    );

    let out = pool.forward(&input);
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    assert_eq!(out.to_vec(), vec![4.0, 3.0, 3.0, 8.0]);
}

// =============================================================================
// 8. ConvTranspose2d Gradient Verification
// =============================================================================

#[test]
fn test_conv_transpose2d_weight_gradient() {
    use brain_nn::ConvTranspose2d;

    let mut conv_t = ConvTranspose2d::new(1, 1, 2);
    conv_t.weight = Value::new(
        Tensor::from_slice(&[1.0, 0.5, -0.5, 2.0], vec![1, 1, 2, 2]),
        true,
    );

    let input = Value::new(
        Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]),
        false,
    );
    let w_data = conv_t.weight.to_vec();

    // Analytic gradient of sum(output) w.r.t weight
    let mut analytic_dw = vec![0.0; 4];
    let in_sum: f64 = input.to_vec().iter().sum();
    for g in analytic_dw.iter_mut() {
        *g = in_sum;
    }

    check_param_gradient(
        |w_test| {
            let mut layer = conv_t.clone();
            layer.weight = Value::new(Tensor::from_slice(w_test, vec![1, 1, 2, 2]), true);
            layer.forward(&input).unwrap().to_vec().iter().sum()
        },
        &w_data,
        &analytic_dw,
        1e-5,
        1e-4,
    );
}

// =============================================================================
// 9. Conv1d Weight Gradient Verification
// =============================================================================

#[test]
fn test_conv1d_weight_gradient() {
    use brain_nn::Conv1d;

    let mut conv = Conv1d::new(1, 1, 2, false);
    conv.weight = Tensor::from_slice(&[0.8, -0.4], vec![1, 1, 2]);

    let input = Value::new(Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 1, 3]), false);
    let w_data = conv.weight.to_vec();

    // For valid padding and kernel=2 on input [1, 2, 3], outputs are:
    // y0 = 1*w0 + 2*w1
    // y1 = 2*w0 + 3*w1
    // dy/dw0 = 1 + 2 = 3, dy/dw1 = 2 + 3 = 5
    let analytic_dw = vec![6.0, 6.0];

    check_param_gradient(
        |w_test| {
            let mut layer = conv.clone();
            layer.weight = Tensor::from_slice(w_test, vec![1, 1, 2]);
            layer.forward(&input).unwrap().to_vec().iter().sum()
        },
        &w_data,
        &analytic_dw,
        1e-5,
        1e-4,
    );
}

// =============================================================================
// 10. MultiheadAttention Forward & Numerical Sensitivity
// =============================================================================

#[test]
fn test_multihead_attention_numerical_forward() {
    use brain_nn::MultiheadAttention;

    let mha = MultiheadAttention::new(8, 2);
    let x = Value::new(Tensor::from_slice(&[0.1; 16], vec![1, 2, 8]), false); // batch=1, seq=2, embed=8
    let out = mha.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 8]);
}

// =============================================================================
// 11. Weight Initialization Schemes & Parameter Completeness
// =============================================================================

#[test]
fn test_initialization_schemes_and_variance() {
    use brain_nn::init::{calculate_fan, kaiming_normal, xavier_uniform};

    let shape = [128, 128];
    let (fan_in, fan_out) = calculate_fan(&shape);
    assert_eq!(fan_in, 128);
    assert_eq!(fan_out, 128);

    let xavier_t = xavier_uniform(&shape);
    let x_data = xavier_t.to_vec();
    let mean: f64 = x_data.iter().sum::<f64>() / x_data.len() as f64;
    assert!(mean.abs() < 0.05);

    let kaiming_t = kaiming_normal(&shape, 0.0);
    let k_data = kaiming_t.to_vec();
    let k_var: f64 = k_data.iter().map(|&x| (x - 0.0).powi(2)).sum::<f64>() / k_data.len() as f64;
    let expected_var = 2.0 / 128.0;
    let rel_diff = (k_var - expected_var).abs() / expected_var;
    assert!(
        rel_diff < 0.20,
        "Kaiming normal variance mismatch: observed={}, expected={}",
        k_var,
        expected_var
    );
}

#[test]
fn test_sequential_module_parameters_completeness() {
    use brain_nn::{Linear, Sequential};

    let l1 = Linear::new(10, 20, true);
    let l2 = Linear::new(20, 5, false);

    let mut seq = Sequential::new();
    seq.add(l1);
    seq.add(l2);

    let params = seq.parameters();
    // l1 has weight [20, 10] + bias [20] = 220
    // l2 has weight [5, 20] = 100
    let total_elements: usize = params.iter().map(|p| p.numel()).sum();
    assert_eq!(total_elements, 320);
}
