//! Fused LSTM forward pass.
//!
//! One implementation, expressed in high-level burn tensor ops. Burn's CubeCL
//! backends fuse the elementwise cell math into their own kernels, so the
//! per-timestep op count no longer needs a hand-written kernel to collapse it.
//!
//! The two matmuls are still hoisted deliberately:
//!
//! 1. The full-sequence input projection `input @ w_ih + bias` runs once,
//!    outside the timestep loop (`burn::nn::Lstm` does four per step).
//! 2. `h_{t-1} @ w_hh` is a single batched matmul rather than four per-gate
//!    linear transforms.

use burn::prelude::*;
use burn::tensor::activation::{sigmoid, tanh};

/// Final hidden / cell state produced by a fused LSTM forward pass.
#[derive(Debug, Clone)]
pub struct FusedLstmStateOut {
    /// Final hidden state. Shape: `[batch, hidden]`.
    pub hidden: Tensor<2>,
    /// Final cell state. Shape: `[batch, hidden]`.
    pub cell: Tensor<2>,
}

/// Run the LSTM forward pass over a full sequence.
///
/// # Arguments
/// - `input`: `[batch, seq_len, input_size]`.
/// - `w_ih`:  `[input_size,  4 * hidden]`, gate order `[i, f, g, o]`.
/// - `w_hh`:  `[hidden,      4 * hidden]`, gate order `[i, f, g, o]`.
/// - `bias`:  optional `[4 * hidden]`.
/// - `h0`:    optional initial hidden state `[batch, hidden]` (zeros if `None`).
/// - `c0`:    optional initial cell state   `[batch, hidden]` (zeros if `None`).
///
/// # Returns
/// - `output`: `[batch, seq_len, hidden]` — hidden state at every step.
/// - `state`:  final hidden and cell state.
pub fn fused_lstm_forward(
    input: Tensor<3>,
    w_ih: Tensor<2>,
    w_hh: Tensor<2>,
    bias: Option<Tensor<1>>,
    h0: Option<Tensor<2>>,
    c0: Option<Tensor<2>>,
) -> (Tensor<3>, FusedLstmStateOut) {
    let [batch, seq_len, _input_size] = input.dims();
    let [_, four_hidden] = w_ih.dims();
    assert!(
        four_hidden.is_multiple_of(4),
        "w_ih last dim must be divisible by 4 (got {four_hidden})"
    );
    let hidden = four_hidden / 4;
    let device = input.device();

    let mut h = h0.unwrap_or_else(|| Tensor::zeros([batch, hidden], &device));
    let mut c = c0.unwrap_or_else(|| Tensor::zeros([batch, hidden], &device));

    // Precompute x @ w_ih + bias over every timestep in one matmul.
    // Shape: [batch, seq_len, 4*hidden].
    let x_proj = input.matmul(w_ih.unsqueeze::<3>());
    let x_proj = match bias.as_ref() {
        Some(b) => x_proj + b.clone().unsqueeze::<3>(),
        None => x_proj,
    };

    let mut outputs: Vec<Tensor<2>> = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let x_t: Tensor<2> = x_proj
            .clone()
            .slice([0..batch, t..(t + 1), 0..four_hidden])
            .reshape([batch, four_hidden]);

        let z = x_t + h.clone().matmul(w_hh.clone());

        let i_gate = sigmoid(z.clone().slice([0..batch, 0..hidden]));
        let f_gate = sigmoid(z.clone().slice([0..batch, hidden..2 * hidden]));
        let g_gate = tanh(z.clone().slice([0..batch, 2 * hidden..3 * hidden]));
        let o_gate = sigmoid(z.slice([0..batch, 3 * hidden..4 * hidden]));

        c = f_gate * c + i_gate * g_gate;
        h = o_gate * tanh(c.clone());

        outputs.push(h.clone());
    }

    let outputs_expanded: Vec<Tensor<3>> =
        outputs.into_iter().map(|t| t.unsqueeze_dim(1)).collect();
    let output = Tensor::cat(outputs_expanded, 1);

    (output, FusedLstmStateOut { hidden: h, cell: c })
}
