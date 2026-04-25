//! Fused LSTM module.
//!
//! [`FusedLstm`] is the only LSTM implementation used in this workspace.
//! It stores weights in a concatenated 4-gate layout (`[i, f, g, o]`,
//! PyTorch / cuDNN convention) and dispatches the forward pass through the
//! [`FusedLstmBackend`] trait:
//!
//! - **NdArray** (CPU / WASM): generic `Tensor`-op loop — correct on every
//!   platform, used for web inference.
//! - **`CubeBackend<R, …>`** (non-WASM builds): one CubeCL cell-kernel
//!   launch per timestep + full-sequence matmuls, used for training on
//!   Wgpu (Windows/Linux/macOS).
//! - **`Autodiff<B>`**: single tracked `Backward` op wrapping
//!   `B::fused_lstm_forward_train`, so the autograd graph stays O(1)
//!   regardless of sequence length.
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

pub(crate) mod backend;
mod module;

#[cfg(not(target_arch = "wasm32"))]
mod cube_kernel;

pub use backend::FusedLstmBackend;
pub use module::{FusedLstm, FusedLstmConfig, FusedLstmState};
