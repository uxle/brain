//! Tests for RNN, LSTM, and GRU sequences
use brain_core::Tensor;
use brain_rnn::*;

#[test]
fn test_lstm_cell_and_gru_cell() {
    let lstm = LstmCell::new(4, 8);
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let h = Tensor::zeros(vec![1, 8]);
    let c = Tensor::zeros(vec![1, 8]);

    let state = CellState::new_lstm(h.clone(), c);
    let (h_next, _) = lstm.forward(&x, &state).unwrap();
    assert_eq!(h_next.shape(), &[1, 8]);

    let gru = GruCell::new(4, 8);
    let gru_state = CellState::new_single(h);
    let (h_gru, _) = gru.forward(&x, &gru_state).unwrap();
    assert_eq!(h_gru.shape(), &[1, 8]);
}

#[test]
fn test_peephole_lstm_and_bidirectional_rnn() {
    // Peephole LSTM
    let peephole = PeepholeLstmCell::new(4, 8);
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let state = peephole.init_state(1);
    let (h_next, new_state) = peephole.forward(&x, &state).unwrap();
    assert_eq!(h_next.shape(), &[1, 8]);
    assert!(matches!(new_state, CellState::Lstm { .. }));

    // Bidirectional LSTM Sequence (Batch=1, Seq=3, Dim=4)
    let bi_rnn = BidirectionalRnn::new(4, 8, 1, BidirectionalMerge::Concat);
    let seq_input = Tensor::ones(vec![1, 3, 4]);
    let bi_out = bi_rnn.forward(&seq_input, None).unwrap();
    assert_eq!(bi_out.output.shape(), &[1, 3, 16]); // Concat: 8 + 8 = 16
}
