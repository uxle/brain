//! Tests for RNN, LSTM, and GRU sequences
use brain_rnn::*;
use brain_core::Tensor;

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
