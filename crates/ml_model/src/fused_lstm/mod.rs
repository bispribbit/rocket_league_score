//! Fused LSTM module.
//!
//! [`FusedLstm`] is the only LSTM implementation used in this workspace.
//! It stores weights in a concatenated 4-gate layout (`[i, f, g, o]`,
//! PyTorch / cuDNN convention) so the gate projections collapse into a
//! single matmul, and runs the cell math in high-level burn tensor ops.
//!
//! Burn's CubeCL backends fuse those elementwise ops into their own kernels,
//! and autodiff tracks the cell math per op, so there is one forward path for
//! every backend and device — CPU, WebGPU, CUDA and WASM alike.
//!
//! ## Layout conventions
//!
//! Input tensor:  `[batch, seq_len, input_size]` (always `batch_first`)
//! Hidden / cell: `[batch, hidden_size]`
//!
//! Weights (PyTorch / cuDNN order, ready for a single fused matmul):
//! - `w_ih`: `[input_size,  4 * hidden_size]`
//! - `w_hh`: `[hidden_size, 4 * hidden_size]`
//! - `bias`: `[4 * hidden_size]` (optional)
//!
//! Gate split order after the matmul: `[i | f | g | o]`, then
//! `i, f, o` go through sigmoid and `g` through tanh.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod backend;
mod forward;
mod module;

pub use forward::{FusedLstmStateOut, fused_lstm_forward};
pub use module::{FusedLstm, FusedLstmConfig, FusedLstmState};
