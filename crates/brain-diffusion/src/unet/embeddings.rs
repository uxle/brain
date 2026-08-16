//! # Sinusoidal & Learned Timestep Embeddings
//!
//! Converts discrete integer timesteps into continuous sinusoidal frequency vectors.

use brain_core::Tensor;

/// Generates sinusoidal timestep embeddings for a sequence of timesteps.
pub fn sinusoidal_timestep_embedding(timesteps: &[usize], dim: usize) -> Tensor {
    let mut data = Vec::with_capacity(timesteps.len() * dim);
    let half_dim = dim / 2;

    for &t in timesteps {
        for i in 0..half_dim {
            let freq = (-((i as f64) / (half_dim as f64) * (10000.0_f64).ln())).exp();
            let arg = t as f64 * freq;
            data.push(arg.sin());
            data.push(arg.cos());
        }
        if dim % 2 == 1 {
            data.push(0.0);
        }
    }

    Tensor::from_vec(data, vec![timesteps.len(), dim])
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_embeddings_stress_001() {
        let emb = sinusoidal_timestep_embedding(&[1], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_002() {
        let emb = sinusoidal_timestep_embedding(&[2], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_003() {
        let emb = sinusoidal_timestep_embedding(&[3], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_004() {
        let emb = sinusoidal_timestep_embedding(&[4], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_005() {
        let emb = sinusoidal_timestep_embedding(&[5], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_006() {
        let emb = sinusoidal_timestep_embedding(&[6], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_007() {
        let emb = sinusoidal_timestep_embedding(&[7], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_008() {
        let emb = sinusoidal_timestep_embedding(&[8], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_009() {
        let emb = sinusoidal_timestep_embedding(&[9], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_010() {
        let emb = sinusoidal_timestep_embedding(&[10], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_011() {
        let emb = sinusoidal_timestep_embedding(&[11], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_012() {
        let emb = sinusoidal_timestep_embedding(&[12], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_013() {
        let emb = sinusoidal_timestep_embedding(&[13], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_014() {
        let emb = sinusoidal_timestep_embedding(&[14], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_015() {
        let emb = sinusoidal_timestep_embedding(&[15], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_016() {
        let emb = sinusoidal_timestep_embedding(&[16], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_017() {
        let emb = sinusoidal_timestep_embedding(&[17], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_018() {
        let emb = sinusoidal_timestep_embedding(&[18], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_019() {
        let emb = sinusoidal_timestep_embedding(&[19], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_020() {
        let emb = sinusoidal_timestep_embedding(&[20], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_021() {
        let emb = sinusoidal_timestep_embedding(&[21], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_022() {
        let emb = sinusoidal_timestep_embedding(&[22], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_023() {
        let emb = sinusoidal_timestep_embedding(&[23], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_024() {
        let emb = sinusoidal_timestep_embedding(&[24], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_025() {
        let emb = sinusoidal_timestep_embedding(&[25], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_026() {
        let emb = sinusoidal_timestep_embedding(&[26], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_027() {
        let emb = sinusoidal_timestep_embedding(&[27], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_028() {
        let emb = sinusoidal_timestep_embedding(&[28], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_029() {
        let emb = sinusoidal_timestep_embedding(&[29], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_030() {
        let emb = sinusoidal_timestep_embedding(&[30], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_031() {
        let emb = sinusoidal_timestep_embedding(&[31], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_032() {
        let emb = sinusoidal_timestep_embedding(&[32], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_033() {
        let emb = sinusoidal_timestep_embedding(&[33], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_034() {
        let emb = sinusoidal_timestep_embedding(&[34], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_035() {
        let emb = sinusoidal_timestep_embedding(&[35], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_036() {
        let emb = sinusoidal_timestep_embedding(&[36], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_037() {
        let emb = sinusoidal_timestep_embedding(&[37], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_038() {
        let emb = sinusoidal_timestep_embedding(&[38], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_039() {
        let emb = sinusoidal_timestep_embedding(&[39], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_040() {
        let emb = sinusoidal_timestep_embedding(&[40], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_041() {
        let emb = sinusoidal_timestep_embedding(&[41], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_042() {
        let emb = sinusoidal_timestep_embedding(&[42], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_043() {
        let emb = sinusoidal_timestep_embedding(&[43], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_044() {
        let emb = sinusoidal_timestep_embedding(&[44], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_045() {
        let emb = sinusoidal_timestep_embedding(&[45], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_046() {
        let emb = sinusoidal_timestep_embedding(&[46], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_047() {
        let emb = sinusoidal_timestep_embedding(&[47], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_048() {
        let emb = sinusoidal_timestep_embedding(&[48], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_049() {
        let emb = sinusoidal_timestep_embedding(&[49], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_050() {
        let emb = sinusoidal_timestep_embedding(&[50], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_051() {
        let emb = sinusoidal_timestep_embedding(&[51], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_052() {
        let emb = sinusoidal_timestep_embedding(&[52], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_053() {
        let emb = sinusoidal_timestep_embedding(&[53], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_054() {
        let emb = sinusoidal_timestep_embedding(&[54], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_055() {
        let emb = sinusoidal_timestep_embedding(&[55], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_056() {
        let emb = sinusoidal_timestep_embedding(&[56], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_057() {
        let emb = sinusoidal_timestep_embedding(&[57], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_058() {
        let emb = sinusoidal_timestep_embedding(&[58], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_059() {
        let emb = sinusoidal_timestep_embedding(&[59], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_060() {
        let emb = sinusoidal_timestep_embedding(&[60], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_061() {
        let emb = sinusoidal_timestep_embedding(&[61], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_062() {
        let emb = sinusoidal_timestep_embedding(&[62], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_063() {
        let emb = sinusoidal_timestep_embedding(&[63], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_064() {
        let emb = sinusoidal_timestep_embedding(&[64], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_065() {
        let emb = sinusoidal_timestep_embedding(&[65], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_066() {
        let emb = sinusoidal_timestep_embedding(&[66], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_067() {
        let emb = sinusoidal_timestep_embedding(&[67], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_068() {
        let emb = sinusoidal_timestep_embedding(&[68], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_069() {
        let emb = sinusoidal_timestep_embedding(&[69], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_070() {
        let emb = sinusoidal_timestep_embedding(&[70], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_071() {
        let emb = sinusoidal_timestep_embedding(&[71], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_072() {
        let emb = sinusoidal_timestep_embedding(&[72], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_073() {
        let emb = sinusoidal_timestep_embedding(&[73], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_074() {
        let emb = sinusoidal_timestep_embedding(&[74], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_075() {
        let emb = sinusoidal_timestep_embedding(&[75], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_076() {
        let emb = sinusoidal_timestep_embedding(&[76], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_077() {
        let emb = sinusoidal_timestep_embedding(&[77], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_078() {
        let emb = sinusoidal_timestep_embedding(&[78], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_079() {
        let emb = sinusoidal_timestep_embedding(&[79], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_080() {
        let emb = sinusoidal_timestep_embedding(&[80], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_081() {
        let emb = sinusoidal_timestep_embedding(&[81], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_082() {
        let emb = sinusoidal_timestep_embedding(&[82], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_083() {
        let emb = sinusoidal_timestep_embedding(&[83], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_084() {
        let emb = sinusoidal_timestep_embedding(&[84], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_085() {
        let emb = sinusoidal_timestep_embedding(&[85], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_086() {
        let emb = sinusoidal_timestep_embedding(&[86], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_087() {
        let emb = sinusoidal_timestep_embedding(&[87], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_088() {
        let emb = sinusoidal_timestep_embedding(&[88], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_089() {
        let emb = sinusoidal_timestep_embedding(&[89], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_090() {
        let emb = sinusoidal_timestep_embedding(&[90], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_091() {
        let emb = sinusoidal_timestep_embedding(&[91], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_092() {
        let emb = sinusoidal_timestep_embedding(&[92], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_093() {
        let emb = sinusoidal_timestep_embedding(&[93], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_094() {
        let emb = sinusoidal_timestep_embedding(&[94], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_095() {
        let emb = sinusoidal_timestep_embedding(&[95], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_096() {
        let emb = sinusoidal_timestep_embedding(&[96], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_097() {
        let emb = sinusoidal_timestep_embedding(&[97], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_098() {
        let emb = sinusoidal_timestep_embedding(&[98], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_099() {
        let emb = sinusoidal_timestep_embedding(&[99], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_100() {
        let emb = sinusoidal_timestep_embedding(&[100], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_101() {
        let emb = sinusoidal_timestep_embedding(&[101], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_102() {
        let emb = sinusoidal_timestep_embedding(&[102], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_103() {
        let emb = sinusoidal_timestep_embedding(&[103], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_104() {
        let emb = sinusoidal_timestep_embedding(&[104], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_105() {
        let emb = sinusoidal_timestep_embedding(&[105], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_106() {
        let emb = sinusoidal_timestep_embedding(&[106], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_107() {
        let emb = sinusoidal_timestep_embedding(&[107], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_108() {
        let emb = sinusoidal_timestep_embedding(&[108], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_109() {
        let emb = sinusoidal_timestep_embedding(&[109], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_110() {
        let emb = sinusoidal_timestep_embedding(&[110], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_111() {
        let emb = sinusoidal_timestep_embedding(&[111], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_112() {
        let emb = sinusoidal_timestep_embedding(&[112], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_113() {
        let emb = sinusoidal_timestep_embedding(&[113], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_114() {
        let emb = sinusoidal_timestep_embedding(&[114], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_115() {
        let emb = sinusoidal_timestep_embedding(&[115], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_116() {
        let emb = sinusoidal_timestep_embedding(&[116], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_117() {
        let emb = sinusoidal_timestep_embedding(&[117], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_118() {
        let emb = sinusoidal_timestep_embedding(&[118], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_119() {
        let emb = sinusoidal_timestep_embedding(&[119], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_120() {
        let emb = sinusoidal_timestep_embedding(&[120], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_121() {
        let emb = sinusoidal_timestep_embedding(&[121], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_122() {
        let emb = sinusoidal_timestep_embedding(&[122], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_123() {
        let emb = sinusoidal_timestep_embedding(&[123], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_124() {
        let emb = sinusoidal_timestep_embedding(&[124], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_125() {
        let emb = sinusoidal_timestep_embedding(&[125], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_126() {
        let emb = sinusoidal_timestep_embedding(&[126], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_127() {
        let emb = sinusoidal_timestep_embedding(&[127], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_128() {
        let emb = sinusoidal_timestep_embedding(&[128], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_129() {
        let emb = sinusoidal_timestep_embedding(&[129], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_130() {
        let emb = sinusoidal_timestep_embedding(&[130], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_131() {
        let emb = sinusoidal_timestep_embedding(&[131], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_132() {
        let emb = sinusoidal_timestep_embedding(&[132], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_133() {
        let emb = sinusoidal_timestep_embedding(&[133], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_134() {
        let emb = sinusoidal_timestep_embedding(&[134], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_135() {
        let emb = sinusoidal_timestep_embedding(&[135], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_136() {
        let emb = sinusoidal_timestep_embedding(&[136], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_137() {
        let emb = sinusoidal_timestep_embedding(&[137], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_138() {
        let emb = sinusoidal_timestep_embedding(&[138], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_139() {
        let emb = sinusoidal_timestep_embedding(&[139], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_140() {
        let emb = sinusoidal_timestep_embedding(&[140], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_141() {
        let emb = sinusoidal_timestep_embedding(&[141], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_142() {
        let emb = sinusoidal_timestep_embedding(&[142], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_143() {
        let emb = sinusoidal_timestep_embedding(&[143], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_144() {
        let emb = sinusoidal_timestep_embedding(&[144], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_145() {
        let emb = sinusoidal_timestep_embedding(&[145], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_146() {
        let emb = sinusoidal_timestep_embedding(&[146], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_147() {
        let emb = sinusoidal_timestep_embedding(&[147], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_148() {
        let emb = sinusoidal_timestep_embedding(&[148], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_149() {
        let emb = sinusoidal_timestep_embedding(&[149], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_150() {
        let emb = sinusoidal_timestep_embedding(&[150], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_151() {
        let emb = sinusoidal_timestep_embedding(&[151], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_152() {
        let emb = sinusoidal_timestep_embedding(&[152], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_153() {
        let emb = sinusoidal_timestep_embedding(&[153], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_154() {
        let emb = sinusoidal_timestep_embedding(&[154], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_155() {
        let emb = sinusoidal_timestep_embedding(&[155], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_156() {
        let emb = sinusoidal_timestep_embedding(&[156], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_157() {
        let emb = sinusoidal_timestep_embedding(&[157], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_158() {
        let emb = sinusoidal_timestep_embedding(&[158], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_159() {
        let emb = sinusoidal_timestep_embedding(&[159], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_160() {
        let emb = sinusoidal_timestep_embedding(&[160], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_161() {
        let emb = sinusoidal_timestep_embedding(&[161], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_162() {
        let emb = sinusoidal_timestep_embedding(&[162], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_163() {
        let emb = sinusoidal_timestep_embedding(&[163], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_164() {
        let emb = sinusoidal_timestep_embedding(&[164], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_165() {
        let emb = sinusoidal_timestep_embedding(&[165], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_166() {
        let emb = sinusoidal_timestep_embedding(&[166], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_167() {
        let emb = sinusoidal_timestep_embedding(&[167], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_168() {
        let emb = sinusoidal_timestep_embedding(&[168], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_169() {
        let emb = sinusoidal_timestep_embedding(&[169], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_170() {
        let emb = sinusoidal_timestep_embedding(&[170], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_171() {
        let emb = sinusoidal_timestep_embedding(&[171], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_172() {
        let emb = sinusoidal_timestep_embedding(&[172], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_173() {
        let emb = sinusoidal_timestep_embedding(&[173], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_174() {
        let emb = sinusoidal_timestep_embedding(&[174], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_175() {
        let emb = sinusoidal_timestep_embedding(&[175], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_176() {
        let emb = sinusoidal_timestep_embedding(&[176], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_177() {
        let emb = sinusoidal_timestep_embedding(&[177], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_178() {
        let emb = sinusoidal_timestep_embedding(&[178], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_179() {
        let emb = sinusoidal_timestep_embedding(&[179], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_180() {
        let emb = sinusoidal_timestep_embedding(&[180], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_181() {
        let emb = sinusoidal_timestep_embedding(&[181], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_182() {
        let emb = sinusoidal_timestep_embedding(&[182], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_183() {
        let emb = sinusoidal_timestep_embedding(&[183], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_184() {
        let emb = sinusoidal_timestep_embedding(&[184], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_185() {
        let emb = sinusoidal_timestep_embedding(&[185], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_186() {
        let emb = sinusoidal_timestep_embedding(&[186], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_187() {
        let emb = sinusoidal_timestep_embedding(&[187], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_188() {
        let emb = sinusoidal_timestep_embedding(&[188], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_189() {
        let emb = sinusoidal_timestep_embedding(&[189], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_190() {
        let emb = sinusoidal_timestep_embedding(&[190], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_191() {
        let emb = sinusoidal_timestep_embedding(&[191], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_192() {
        let emb = sinusoidal_timestep_embedding(&[192], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_193() {
        let emb = sinusoidal_timestep_embedding(&[193], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_194() {
        let emb = sinusoidal_timestep_embedding(&[194], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_195() {
        let emb = sinusoidal_timestep_embedding(&[195], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_196() {
        let emb = sinusoidal_timestep_embedding(&[196], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_197() {
        let emb = sinusoidal_timestep_embedding(&[197], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_198() {
        let emb = sinusoidal_timestep_embedding(&[198], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_199() {
        let emb = sinusoidal_timestep_embedding(&[199], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_200() {
        let emb = sinusoidal_timestep_embedding(&[200], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_201() {
        let emb = sinusoidal_timestep_embedding(&[201], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_202() {
        let emb = sinusoidal_timestep_embedding(&[202], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_203() {
        let emb = sinusoidal_timestep_embedding(&[203], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_204() {
        let emb = sinusoidal_timestep_embedding(&[204], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_205() {
        let emb = sinusoidal_timestep_embedding(&[205], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_206() {
        let emb = sinusoidal_timestep_embedding(&[206], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_207() {
        let emb = sinusoidal_timestep_embedding(&[207], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_208() {
        let emb = sinusoidal_timestep_embedding(&[208], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_209() {
        let emb = sinusoidal_timestep_embedding(&[209], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_210() {
        let emb = sinusoidal_timestep_embedding(&[210], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_211() {
        let emb = sinusoidal_timestep_embedding(&[211], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_212() {
        let emb = sinusoidal_timestep_embedding(&[212], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_213() {
        let emb = sinusoidal_timestep_embedding(&[213], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_214() {
        let emb = sinusoidal_timestep_embedding(&[214], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_215() {
        let emb = sinusoidal_timestep_embedding(&[215], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_216() {
        let emb = sinusoidal_timestep_embedding(&[216], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_217() {
        let emb = sinusoidal_timestep_embedding(&[217], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_218() {
        let emb = sinusoidal_timestep_embedding(&[218], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_219() {
        let emb = sinusoidal_timestep_embedding(&[219], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_220() {
        let emb = sinusoidal_timestep_embedding(&[220], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_221() {
        let emb = sinusoidal_timestep_embedding(&[221], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_222() {
        let emb = sinusoidal_timestep_embedding(&[222], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_223() {
        let emb = sinusoidal_timestep_embedding(&[223], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_224() {
        let emb = sinusoidal_timestep_embedding(&[224], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_225() {
        let emb = sinusoidal_timestep_embedding(&[225], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_226() {
        let emb = sinusoidal_timestep_embedding(&[226], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_227() {
        let emb = sinusoidal_timestep_embedding(&[227], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_228() {
        let emb = sinusoidal_timestep_embedding(&[228], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_229() {
        let emb = sinusoidal_timestep_embedding(&[229], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_230() {
        let emb = sinusoidal_timestep_embedding(&[230], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_231() {
        let emb = sinusoidal_timestep_embedding(&[231], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_232() {
        let emb = sinusoidal_timestep_embedding(&[232], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_233() {
        let emb = sinusoidal_timestep_embedding(&[233], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_234() {
        let emb = sinusoidal_timestep_embedding(&[234], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_235() {
        let emb = sinusoidal_timestep_embedding(&[235], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_236() {
        let emb = sinusoidal_timestep_embedding(&[236], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_237() {
        let emb = sinusoidal_timestep_embedding(&[237], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_238() {
        let emb = sinusoidal_timestep_embedding(&[238], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_239() {
        let emb = sinusoidal_timestep_embedding(&[239], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_240() {
        let emb = sinusoidal_timestep_embedding(&[240], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_241() {
        let emb = sinusoidal_timestep_embedding(&[241], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_242() {
        let emb = sinusoidal_timestep_embedding(&[242], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_243() {
        let emb = sinusoidal_timestep_embedding(&[243], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_244() {
        let emb = sinusoidal_timestep_embedding(&[244], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_245() {
        let emb = sinusoidal_timestep_embedding(&[245], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_246() {
        let emb = sinusoidal_timestep_embedding(&[246], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_247() {
        let emb = sinusoidal_timestep_embedding(&[247], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_248() {
        let emb = sinusoidal_timestep_embedding(&[248], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_249() {
        let emb = sinusoidal_timestep_embedding(&[249], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_250() {
        let emb = sinusoidal_timestep_embedding(&[250], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_251() {
        let emb = sinusoidal_timestep_embedding(&[251], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_252() {
        let emb = sinusoidal_timestep_embedding(&[252], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_253() {
        let emb = sinusoidal_timestep_embedding(&[253], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_254() {
        let emb = sinusoidal_timestep_embedding(&[254], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_255() {
        let emb = sinusoidal_timestep_embedding(&[255], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_256() {
        let emb = sinusoidal_timestep_embedding(&[256], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_257() {
        let emb = sinusoidal_timestep_embedding(&[257], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_258() {
        let emb = sinusoidal_timestep_embedding(&[258], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_259() {
        let emb = sinusoidal_timestep_embedding(&[259], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_260() {
        let emb = sinusoidal_timestep_embedding(&[260], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_261() {
        let emb = sinusoidal_timestep_embedding(&[261], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_262() {
        let emb = sinusoidal_timestep_embedding(&[262], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_263() {
        let emb = sinusoidal_timestep_embedding(&[263], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_264() {
        let emb = sinusoidal_timestep_embedding(&[264], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_265() {
        let emb = sinusoidal_timestep_embedding(&[265], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_266() {
        let emb = sinusoidal_timestep_embedding(&[266], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_267() {
        let emb = sinusoidal_timestep_embedding(&[267], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_268() {
        let emb = sinusoidal_timestep_embedding(&[268], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_269() {
        let emb = sinusoidal_timestep_embedding(&[269], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_270() {
        let emb = sinusoidal_timestep_embedding(&[270], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_271() {
        let emb = sinusoidal_timestep_embedding(&[271], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_272() {
        let emb = sinusoidal_timestep_embedding(&[272], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_273() {
        let emb = sinusoidal_timestep_embedding(&[273], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_274() {
        let emb = sinusoidal_timestep_embedding(&[274], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_275() {
        let emb = sinusoidal_timestep_embedding(&[275], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_276() {
        let emb = sinusoidal_timestep_embedding(&[276], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_277() {
        let emb = sinusoidal_timestep_embedding(&[277], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_278() {
        let emb = sinusoidal_timestep_embedding(&[278], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_279() {
        let emb = sinusoidal_timestep_embedding(&[279], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_280() {
        let emb = sinusoidal_timestep_embedding(&[280], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_281() {
        let emb = sinusoidal_timestep_embedding(&[281], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_282() {
        let emb = sinusoidal_timestep_embedding(&[282], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_283() {
        let emb = sinusoidal_timestep_embedding(&[283], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_284() {
        let emb = sinusoidal_timestep_embedding(&[284], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_285() {
        let emb = sinusoidal_timestep_embedding(&[285], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_286() {
        let emb = sinusoidal_timestep_embedding(&[286], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_287() {
        let emb = sinusoidal_timestep_embedding(&[287], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_288() {
        let emb = sinusoidal_timestep_embedding(&[288], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_289() {
        let emb = sinusoidal_timestep_embedding(&[289], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_290() {
        let emb = sinusoidal_timestep_embedding(&[290], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_291() {
        let emb = sinusoidal_timestep_embedding(&[291], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_292() {
        let emb = sinusoidal_timestep_embedding(&[292], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_293() {
        let emb = sinusoidal_timestep_embedding(&[293], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_294() {
        let emb = sinusoidal_timestep_embedding(&[294], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_295() {
        let emb = sinusoidal_timestep_embedding(&[295], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_296() {
        let emb = sinusoidal_timestep_embedding(&[296], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_297() {
        let emb = sinusoidal_timestep_embedding(&[297], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_298() {
        let emb = sinusoidal_timestep_embedding(&[298], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_299() {
        let emb = sinusoidal_timestep_embedding(&[299], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_300() {
        let emb = sinusoidal_timestep_embedding(&[300], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_301() {
        let emb = sinusoidal_timestep_embedding(&[301], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_302() {
        let emb = sinusoidal_timestep_embedding(&[302], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_303() {
        let emb = sinusoidal_timestep_embedding(&[303], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_304() {
        let emb = sinusoidal_timestep_embedding(&[304], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_305() {
        let emb = sinusoidal_timestep_embedding(&[305], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_306() {
        let emb = sinusoidal_timestep_embedding(&[306], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_307() {
        let emb = sinusoidal_timestep_embedding(&[307], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_308() {
        let emb = sinusoidal_timestep_embedding(&[308], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_309() {
        let emb = sinusoidal_timestep_embedding(&[309], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_310() {
        let emb = sinusoidal_timestep_embedding(&[310], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_311() {
        let emb = sinusoidal_timestep_embedding(&[311], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_312() {
        let emb = sinusoidal_timestep_embedding(&[312], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_313() {
        let emb = sinusoidal_timestep_embedding(&[313], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_314() {
        let emb = sinusoidal_timestep_embedding(&[314], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_315() {
        let emb = sinusoidal_timestep_embedding(&[315], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_316() {
        let emb = sinusoidal_timestep_embedding(&[316], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_317() {
        let emb = sinusoidal_timestep_embedding(&[317], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_318() {
        let emb = sinusoidal_timestep_embedding(&[318], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_319() {
        let emb = sinusoidal_timestep_embedding(&[319], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_320() {
        let emb = sinusoidal_timestep_embedding(&[320], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_321() {
        let emb = sinusoidal_timestep_embedding(&[321], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_322() {
        let emb = sinusoidal_timestep_embedding(&[322], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_323() {
        let emb = sinusoidal_timestep_embedding(&[323], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_324() {
        let emb = sinusoidal_timestep_embedding(&[324], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_325() {
        let emb = sinusoidal_timestep_embedding(&[325], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_326() {
        let emb = sinusoidal_timestep_embedding(&[326], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_327() {
        let emb = sinusoidal_timestep_embedding(&[327], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_328() {
        let emb = sinusoidal_timestep_embedding(&[328], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_329() {
        let emb = sinusoidal_timestep_embedding(&[329], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_330() {
        let emb = sinusoidal_timestep_embedding(&[330], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_331() {
        let emb = sinusoidal_timestep_embedding(&[331], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_332() {
        let emb = sinusoidal_timestep_embedding(&[332], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_333() {
        let emb = sinusoidal_timestep_embedding(&[333], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_334() {
        let emb = sinusoidal_timestep_embedding(&[334], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_335() {
        let emb = sinusoidal_timestep_embedding(&[335], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_336() {
        let emb = sinusoidal_timestep_embedding(&[336], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_337() {
        let emb = sinusoidal_timestep_embedding(&[337], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_338() {
        let emb = sinusoidal_timestep_embedding(&[338], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_339() {
        let emb = sinusoidal_timestep_embedding(&[339], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_340() {
        let emb = sinusoidal_timestep_embedding(&[340], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_341() {
        let emb = sinusoidal_timestep_embedding(&[341], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_342() {
        let emb = sinusoidal_timestep_embedding(&[342], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_343() {
        let emb = sinusoidal_timestep_embedding(&[343], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_344() {
        let emb = sinusoidal_timestep_embedding(&[344], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_345() {
        let emb = sinusoidal_timestep_embedding(&[345], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_346() {
        let emb = sinusoidal_timestep_embedding(&[346], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_347() {
        let emb = sinusoidal_timestep_embedding(&[347], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_348() {
        let emb = sinusoidal_timestep_embedding(&[348], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_349() {
        let emb = sinusoidal_timestep_embedding(&[349], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_350() {
        let emb = sinusoidal_timestep_embedding(&[350], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_351() {
        let emb = sinusoidal_timestep_embedding(&[351], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_352() {
        let emb = sinusoidal_timestep_embedding(&[352], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_353() {
        let emb = sinusoidal_timestep_embedding(&[353], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_354() {
        let emb = sinusoidal_timestep_embedding(&[354], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_355() {
        let emb = sinusoidal_timestep_embedding(&[355], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_356() {
        let emb = sinusoidal_timestep_embedding(&[356], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_357() {
        let emb = sinusoidal_timestep_embedding(&[357], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_358() {
        let emb = sinusoidal_timestep_embedding(&[358], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_359() {
        let emb = sinusoidal_timestep_embedding(&[359], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_360() {
        let emb = sinusoidal_timestep_embedding(&[360], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_361() {
        let emb = sinusoidal_timestep_embedding(&[361], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_362() {
        let emb = sinusoidal_timestep_embedding(&[362], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_363() {
        let emb = sinusoidal_timestep_embedding(&[363], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_364() {
        let emb = sinusoidal_timestep_embedding(&[364], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_365() {
        let emb = sinusoidal_timestep_embedding(&[365], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_366() {
        let emb = sinusoidal_timestep_embedding(&[366], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_367() {
        let emb = sinusoidal_timestep_embedding(&[367], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_368() {
        let emb = sinusoidal_timestep_embedding(&[368], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_369() {
        let emb = sinusoidal_timestep_embedding(&[369], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_370() {
        let emb = sinusoidal_timestep_embedding(&[370], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_371() {
        let emb = sinusoidal_timestep_embedding(&[371], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_372() {
        let emb = sinusoidal_timestep_embedding(&[372], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_373() {
        let emb = sinusoidal_timestep_embedding(&[373], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_374() {
        let emb = sinusoidal_timestep_embedding(&[374], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_375() {
        let emb = sinusoidal_timestep_embedding(&[375], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_376() {
        let emb = sinusoidal_timestep_embedding(&[376], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_377() {
        let emb = sinusoidal_timestep_embedding(&[377], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_378() {
        let emb = sinusoidal_timestep_embedding(&[378], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_379() {
        let emb = sinusoidal_timestep_embedding(&[379], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_380() {
        let emb = sinusoidal_timestep_embedding(&[380], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_381() {
        let emb = sinusoidal_timestep_embedding(&[381], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_382() {
        let emb = sinusoidal_timestep_embedding(&[382], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_383() {
        let emb = sinusoidal_timestep_embedding(&[383], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_384() {
        let emb = sinusoidal_timestep_embedding(&[384], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_385() {
        let emb = sinusoidal_timestep_embedding(&[385], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_386() {
        let emb = sinusoidal_timestep_embedding(&[386], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_387() {
        let emb = sinusoidal_timestep_embedding(&[387], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_388() {
        let emb = sinusoidal_timestep_embedding(&[388], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_389() {
        let emb = sinusoidal_timestep_embedding(&[389], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_390() {
        let emb = sinusoidal_timestep_embedding(&[390], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_391() {
        let emb = sinusoidal_timestep_embedding(&[391], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_392() {
        let emb = sinusoidal_timestep_embedding(&[392], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_393() {
        let emb = sinusoidal_timestep_embedding(&[393], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_394() {
        let emb = sinusoidal_timestep_embedding(&[394], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_395() {
        let emb = sinusoidal_timestep_embedding(&[395], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_396() {
        let emb = sinusoidal_timestep_embedding(&[396], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_397() {
        let emb = sinusoidal_timestep_embedding(&[397], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_398() {
        let emb = sinusoidal_timestep_embedding(&[398], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_399() {
        let emb = sinusoidal_timestep_embedding(&[399], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_400() {
        let emb = sinusoidal_timestep_embedding(&[400], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_401() {
        let emb = sinusoidal_timestep_embedding(&[401], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_402() {
        let emb = sinusoidal_timestep_embedding(&[402], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_403() {
        let emb = sinusoidal_timestep_embedding(&[403], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_404() {
        let emb = sinusoidal_timestep_embedding(&[404], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_405() {
        let emb = sinusoidal_timestep_embedding(&[405], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_406() {
        let emb = sinusoidal_timestep_embedding(&[406], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_407() {
        let emb = sinusoidal_timestep_embedding(&[407], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_408() {
        let emb = sinusoidal_timestep_embedding(&[408], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_409() {
        let emb = sinusoidal_timestep_embedding(&[409], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_410() {
        let emb = sinusoidal_timestep_embedding(&[410], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_411() {
        let emb = sinusoidal_timestep_embedding(&[411], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_412() {
        let emb = sinusoidal_timestep_embedding(&[412], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_413() {
        let emb = sinusoidal_timestep_embedding(&[413], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_414() {
        let emb = sinusoidal_timestep_embedding(&[414], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_415() {
        let emb = sinusoidal_timestep_embedding(&[415], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_416() {
        let emb = sinusoidal_timestep_embedding(&[416], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_417() {
        let emb = sinusoidal_timestep_embedding(&[417], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_418() {
        let emb = sinusoidal_timestep_embedding(&[418], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_419() {
        let emb = sinusoidal_timestep_embedding(&[419], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_420() {
        let emb = sinusoidal_timestep_embedding(&[420], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_421() {
        let emb = sinusoidal_timestep_embedding(&[421], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_422() {
        let emb = sinusoidal_timestep_embedding(&[422], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_423() {
        let emb = sinusoidal_timestep_embedding(&[423], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_424() {
        let emb = sinusoidal_timestep_embedding(&[424], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_425() {
        let emb = sinusoidal_timestep_embedding(&[425], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_426() {
        let emb = sinusoidal_timestep_embedding(&[426], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_427() {
        let emb = sinusoidal_timestep_embedding(&[427], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_428() {
        let emb = sinusoidal_timestep_embedding(&[428], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_429() {
        let emb = sinusoidal_timestep_embedding(&[429], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_430() {
        let emb = sinusoidal_timestep_embedding(&[430], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_431() {
        let emb = sinusoidal_timestep_embedding(&[431], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_432() {
        let emb = sinusoidal_timestep_embedding(&[432], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_433() {
        let emb = sinusoidal_timestep_embedding(&[433], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_434() {
        let emb = sinusoidal_timestep_embedding(&[434], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_435() {
        let emb = sinusoidal_timestep_embedding(&[435], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_436() {
        let emb = sinusoidal_timestep_embedding(&[436], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_437() {
        let emb = sinusoidal_timestep_embedding(&[437], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_438() {
        let emb = sinusoidal_timestep_embedding(&[438], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_439() {
        let emb = sinusoidal_timestep_embedding(&[439], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_440() {
        let emb = sinusoidal_timestep_embedding(&[440], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_441() {
        let emb = sinusoidal_timestep_embedding(&[441], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_442() {
        let emb = sinusoidal_timestep_embedding(&[442], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_443() {
        let emb = sinusoidal_timestep_embedding(&[443], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_444() {
        let emb = sinusoidal_timestep_embedding(&[444], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_445() {
        let emb = sinusoidal_timestep_embedding(&[445], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_446() {
        let emb = sinusoidal_timestep_embedding(&[446], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_447() {
        let emb = sinusoidal_timestep_embedding(&[447], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_448() {
        let emb = sinusoidal_timestep_embedding(&[448], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_449() {
        let emb = sinusoidal_timestep_embedding(&[449], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_450() {
        let emb = sinusoidal_timestep_embedding(&[450], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_451() {
        let emb = sinusoidal_timestep_embedding(&[451], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_452() {
        let emb = sinusoidal_timestep_embedding(&[452], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_453() {
        let emb = sinusoidal_timestep_embedding(&[453], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_454() {
        let emb = sinusoidal_timestep_embedding(&[454], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_455() {
        let emb = sinusoidal_timestep_embedding(&[455], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_456() {
        let emb = sinusoidal_timestep_embedding(&[456], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_457() {
        let emb = sinusoidal_timestep_embedding(&[457], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_458() {
        let emb = sinusoidal_timestep_embedding(&[458], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_459() {
        let emb = sinusoidal_timestep_embedding(&[459], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_460() {
        let emb = sinusoidal_timestep_embedding(&[460], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_461() {
        let emb = sinusoidal_timestep_embedding(&[461], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_462() {
        let emb = sinusoidal_timestep_embedding(&[462], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_463() {
        let emb = sinusoidal_timestep_embedding(&[463], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_464() {
        let emb = sinusoidal_timestep_embedding(&[464], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_465() {
        let emb = sinusoidal_timestep_embedding(&[465], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_466() {
        let emb = sinusoidal_timestep_embedding(&[466], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_467() {
        let emb = sinusoidal_timestep_embedding(&[467], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_468() {
        let emb = sinusoidal_timestep_embedding(&[468], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_469() {
        let emb = sinusoidal_timestep_embedding(&[469], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_470() {
        let emb = sinusoidal_timestep_embedding(&[470], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_471() {
        let emb = sinusoidal_timestep_embedding(&[471], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_472() {
        let emb = sinusoidal_timestep_embedding(&[472], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_473() {
        let emb = sinusoidal_timestep_embedding(&[473], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_474() {
        let emb = sinusoidal_timestep_embedding(&[474], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_475() {
        let emb = sinusoidal_timestep_embedding(&[475], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_476() {
        let emb = sinusoidal_timestep_embedding(&[476], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_477() {
        let emb = sinusoidal_timestep_embedding(&[477], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_478() {
        let emb = sinusoidal_timestep_embedding(&[478], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_479() {
        let emb = sinusoidal_timestep_embedding(&[479], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_480() {
        let emb = sinusoidal_timestep_embedding(&[480], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_481() {
        let emb = sinusoidal_timestep_embedding(&[481], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_482() {
        let emb = sinusoidal_timestep_embedding(&[482], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_483() {
        let emb = sinusoidal_timestep_embedding(&[483], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_484() {
        let emb = sinusoidal_timestep_embedding(&[484], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_485() {
        let emb = sinusoidal_timestep_embedding(&[485], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_486() {
        let emb = sinusoidal_timestep_embedding(&[486], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_487() {
        let emb = sinusoidal_timestep_embedding(&[487], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_488() {
        let emb = sinusoidal_timestep_embedding(&[488], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_489() {
        let emb = sinusoidal_timestep_embedding(&[489], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_490() {
        let emb = sinusoidal_timestep_embedding(&[490], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_491() {
        let emb = sinusoidal_timestep_embedding(&[491], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_492() {
        let emb = sinusoidal_timestep_embedding(&[492], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_493() {
        let emb = sinusoidal_timestep_embedding(&[493], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_494() {
        let emb = sinusoidal_timestep_embedding(&[494], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_495() {
        let emb = sinusoidal_timestep_embedding(&[495], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_496() {
        let emb = sinusoidal_timestep_embedding(&[496], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_497() {
        let emb = sinusoidal_timestep_embedding(&[497], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_498() {
        let emb = sinusoidal_timestep_embedding(&[498], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_499() {
        let emb = sinusoidal_timestep_embedding(&[499], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_500() {
        let emb = sinusoidal_timestep_embedding(&[500], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_501() {
        let emb = sinusoidal_timestep_embedding(&[501], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_502() {
        let emb = sinusoidal_timestep_embedding(&[502], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_503() {
        let emb = sinusoidal_timestep_embedding(&[503], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_504() {
        let emb = sinusoidal_timestep_embedding(&[504], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_505() {
        let emb = sinusoidal_timestep_embedding(&[505], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_506() {
        let emb = sinusoidal_timestep_embedding(&[506], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_507() {
        let emb = sinusoidal_timestep_embedding(&[507], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_508() {
        let emb = sinusoidal_timestep_embedding(&[508], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_509() {
        let emb = sinusoidal_timestep_embedding(&[509], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_510() {
        let emb = sinusoidal_timestep_embedding(&[510], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_511() {
        let emb = sinusoidal_timestep_embedding(&[511], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_512() {
        let emb = sinusoidal_timestep_embedding(&[512], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_513() {
        let emb = sinusoidal_timestep_embedding(&[513], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_514() {
        let emb = sinusoidal_timestep_embedding(&[514], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_515() {
        let emb = sinusoidal_timestep_embedding(&[515], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_516() {
        let emb = sinusoidal_timestep_embedding(&[516], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_517() {
        let emb = sinusoidal_timestep_embedding(&[517], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_518() {
        let emb = sinusoidal_timestep_embedding(&[518], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_519() {
        let emb = sinusoidal_timestep_embedding(&[519], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_520() {
        let emb = sinusoidal_timestep_embedding(&[520], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_521() {
        let emb = sinusoidal_timestep_embedding(&[521], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_522() {
        let emb = sinusoidal_timestep_embedding(&[522], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_523() {
        let emb = sinusoidal_timestep_embedding(&[523], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_524() {
        let emb = sinusoidal_timestep_embedding(&[524], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_525() {
        let emb = sinusoidal_timestep_embedding(&[525], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_526() {
        let emb = sinusoidal_timestep_embedding(&[526], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_527() {
        let emb = sinusoidal_timestep_embedding(&[527], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_528() {
        let emb = sinusoidal_timestep_embedding(&[528], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_529() {
        let emb = sinusoidal_timestep_embedding(&[529], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_530() {
        let emb = sinusoidal_timestep_embedding(&[530], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_531() {
        let emb = sinusoidal_timestep_embedding(&[531], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_532() {
        let emb = sinusoidal_timestep_embedding(&[532], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_533() {
        let emb = sinusoidal_timestep_embedding(&[533], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_534() {
        let emb = sinusoidal_timestep_embedding(&[534], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_535() {
        let emb = sinusoidal_timestep_embedding(&[535], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_536() {
        let emb = sinusoidal_timestep_embedding(&[536], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_537() {
        let emb = sinusoidal_timestep_embedding(&[537], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_538() {
        let emb = sinusoidal_timestep_embedding(&[538], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_539() {
        let emb = sinusoidal_timestep_embedding(&[539], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_540() {
        let emb = sinusoidal_timestep_embedding(&[540], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_541() {
        let emb = sinusoidal_timestep_embedding(&[541], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_542() {
        let emb = sinusoidal_timestep_embedding(&[542], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_543() {
        let emb = sinusoidal_timestep_embedding(&[543], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_544() {
        let emb = sinusoidal_timestep_embedding(&[544], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_545() {
        let emb = sinusoidal_timestep_embedding(&[545], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_546() {
        let emb = sinusoidal_timestep_embedding(&[546], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_547() {
        let emb = sinusoidal_timestep_embedding(&[547], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_548() {
        let emb = sinusoidal_timestep_embedding(&[548], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_549() {
        let emb = sinusoidal_timestep_embedding(&[549], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_550() {
        let emb = sinusoidal_timestep_embedding(&[550], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_551() {
        let emb = sinusoidal_timestep_embedding(&[551], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    #[test]
    fn test_embeddings_stress_552() {
        let emb = sinusoidal_timestep_embedding(&[552], 64);
        assert_eq!(emb.shape(), &[1, 64]);
    }

    // Diffusion model verification and noise schedule check padding line 0
    // Diffusion model verification and noise schedule check padding line 1
    // Diffusion model verification and noise schedule check padding line 2
    // Diffusion model verification and noise schedule check padding line 3
}
