// The `#[cube(launch)]` macro expands into code that trips several clippy
// lints (format_push_string, trivially_copy_pass_by_ref, redundant_closure,
// manual_range_contains, …). Silencing at the module level keeps us from
// peppering every kernel with individual allows.
#![allow(
    clippy::format_push_string,
    clippy::trivially_copy_pass_by_ref,
    clippy::unnecessary_cast,
    clippy::manual_range_contains,
    clippy::redundant_closure_for_method_calls,
    clippy::needless_pass_by_value,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::used_underscore_binding,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]

//! CubeCL-backed fused LSTM forward pass for `CubeBackend<R, …>`.
//!
//! This replaces the generic `Tensor`-op timestep loop with a single
//! hand-written kernel launched once per timestep. The kernel reads the
//! precomputed input projection and hidden projection, runs the 4-gate
//! activations, updates the cell state, and writes `h_next` + `c_next`.
//!
//! ## Dispatch layout
//!
//! One thread per `(batch, hidden)` output position. Each thread:
//!
//! 1. Reads its 4 gate contributions from `x_proj[b, t, *, h]`.
//! 2. Adds its 4 gate contributions from `(h_prev @ W_hh)[b, *, h]`
//!    (computed by burn's matmul into a separate buffer before launch).
//! 3. Applies `sigmoid` / `tanh` and updates cell + hidden state.
//!
//! Running `h_prev @ W_hh` outside the kernel lets us reuse burn's already
//! hyper-tuned matmul implementation; the cell kernel itself stays small
//! and collapses the ~18 remaining per-timestep elementwise ops
//! (`burn-nn` dispatches 4 linear + 4 add + 4 sigmoid/tanh + update) into
//! a single kernel launch.

use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
use cubecl::prelude::*;
use cubecl::{CubeDim, calculate_cube_count_elemwise};

use super::backend::{FusedLstmBackend, FusedLstmStateOut, FusedLstmTrainOut};

// ───────────────────────────────────────────────────────────────────────────
// Kernel
// ───────────────────────────────────────────────────────────────────────────

/// One fused LSTM cell update for a single timestep.
///
/// All tensors are row-major, contiguous, `[batch, …]`-first.
///
/// - `x_proj_t`: `[batch, 4*hidden]` — `(input_t @ W_ih + b)` for this step.
/// - `h_proj`:   `[batch, 4*hidden]` — `(h_prev @ W_hh)` for this step.
/// - `c_prev`:   `[batch, hidden]`.
/// - `h_next`:   `[batch, hidden]` — written by the kernel.
/// - `c_next`:   `[batch, hidden]` — written by the kernel.
///
/// Gate order along the last dim: `[i, f, g, o]`.
#[cube(launch)]
fn lstm_cell_kernel<F: Float>(
    x_proj_t: &Tensor<F>,
    h_proj: &Tensor<F>,
    c_prev: &Tensor<F>,
    h_next: &mut Tensor<F>,
    c_next: &mut Tensor<F>,
) {
    let hidden = h_next.shape(1);
    let total = h_next.shape(0) * hidden;
    let pos = ABSOLUTE_POS;
    if pos >= total {
        terminate!();
    }

    let b = pos / hidden;
    let h = pos % hidden;

    let four_hidden = 4 * hidden;
    let proj_base = b * four_hidden;

    let z_i = x_proj_t[proj_base + h] + h_proj[proj_base + h];
    let z_f = x_proj_t[proj_base + hidden + h] + h_proj[proj_base + hidden + h];
    let z_g = x_proj_t[proj_base + 2 * hidden + h] + h_proj[proj_base + 2 * hidden + h];
    let z_o = x_proj_t[proj_base + 3 * hidden + h] + h_proj[proj_base + 3 * hidden + h];

    let one = F::new(1.0);
    let gate_i = one / (one + F::exp(-z_i));
    let gate_f = one / (one + F::exp(-z_f));
    let gate_g = F::tanh(z_g);
    let gate_o = one / (one + F::exp(-z_o));

    let c_idx = b * hidden + h;
    let c_new = gate_f * c_prev[c_idx] + gate_i * gate_g;
    let h_new = gate_o * F::tanh(c_new);

    c_next[c_idx] = c_new;
    h_next[c_idx] = h_new;
}

/// Train-aware variant that reads the gate pre-activation `z` already combined
/// (i.e. `x_proj_t + h_proj`) so the caller can save it for BPTT without
/// recomputing it inside the kernel.
///
/// Semantically identical to [`lstm_cell_kernel`]; the only reason two
/// kernels exist is to keep the inference path at one dispatch per step
/// (matmul + kernel). Training needs `z` saved anyway, so its extra
/// `x_proj + h_proj` add happens once, outside.
#[cube(launch)]
fn lstm_cell_kernel_train<F: Float>(
    z: &Tensor<F>,
    c_prev: &Tensor<F>,
    h_next: &mut Tensor<F>,
    c_next: &mut Tensor<F>,
) {
    let hidden = h_next.shape(1);
    let total = h_next.shape(0) * hidden;
    let pos = ABSOLUTE_POS;
    if pos >= total {
        terminate!();
    }

    let b = pos / hidden;
    let h = pos % hidden;

    let four_hidden = 4 * hidden;
    let z_base = b * four_hidden;

    let z_i = z[z_base + h];
    let z_f = z[z_base + hidden + h];
    let z_g = z[z_base + 2 * hidden + h];
    let z_o = z[z_base + 3 * hidden + h];

    let one = F::new(1.0);
    let gate_i = one / (one + F::exp(-z_i));
    let gate_f = one / (one + F::exp(-z_f));
    let gate_g = F::tanh(z_g);
    let gate_o = one / (one + F::exp(-z_o));

    let c_idx = b * hidden + h;
    let c_new = gate_f * c_prev[c_idx] + gate_i * gate_g;
    let h_new = gate_o * F::tanh(c_new);

    c_next[c_idx] = c_new;
    h_next[c_idx] = h_new;
}

// ───────────────────────────────────────────────────────────────────────────
// Launch helpers
// ───────────────────────────────────────────────────────────────────────────

/// Launch [`lstm_cell_kernel`] for one timestep. Allocates fresh `h_next`
/// and `c_next` buffers and returns them.
fn launch_cell<R: CubeRuntime, F: FloatElement>(
    x_proj_t: CubeTensor<R>,
    h_proj: CubeTensor<R>,
    c_prev: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let dtype = c_prev.dtype;
    let client = c_prev.client.clone();
    let device = c_prev.device.clone();

    let [batch, hidden] = c_prev.shape.dims();
    let total = batch * hidden;

    let h_next = empty_device_dtype(client.clone(), device.clone(), c_prev.shape.clone(), dtype);
    let c_next = empty_device_dtype(client.clone(), device, c_prev.shape.clone(), dtype);

    let cube_dim = CubeDim::new(&client, total);
    let cube_count = calculate_cube_count_elemwise(&client, total, cube_dim);

    lstm_cell_kernel::launch::<F, R>(
        &client,
        cube_count,
        cube_dim,
        x_proj_t.as_tensor_arg(1),
        h_proj.as_tensor_arg(1),
        c_prev.as_tensor_arg(1),
        h_next.as_tensor_arg(1),
        c_next.as_tensor_arg(1),
    )
    .expect("lstm_cell_kernel launch failed");

    (h_next, c_next)
}

/// Launch [`lstm_cell_kernel_train`] for one timestep.
fn launch_cell_train<R: CubeRuntime, F: FloatElement>(
    z: CubeTensor<R>,
    c_prev: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let dtype = c_prev.dtype;
    let client = c_prev.client.clone();
    let device = c_prev.device.clone();

    let [batch, hidden] = c_prev.shape.dims();
    let total = batch * hidden;

    let h_next = empty_device_dtype(client.clone(), device.clone(), c_prev.shape.clone(), dtype);
    let c_next = empty_device_dtype(client.clone(), device, c_prev.shape.clone(), dtype);

    let cube_dim = CubeDim::new(&client, total);
    let cube_count = calculate_cube_count_elemwise(&client, total, cube_dim);

    lstm_cell_kernel_train::launch::<F, R>(
        &client,
        cube_count,
        cube_dim,
        z.as_tensor_arg(1),
        c_prev.as_tensor_arg(1),
        h_next.as_tensor_arg(1),
        c_next.as_tensor_arg(1),
    )
    .expect("lstm_cell_kernel_train launch failed");

    (h_next, c_next)
}

// ───────────────────────────────────────────────────────────────────────────
// FusedLstmBackend impl for CubeBackend<R, F, I, B>
// ───────────────────────────────────────────────────────────────────────────

impl<R, F, I, B> FusedLstmBackend for CubeBackend<R, F, I, B>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    B: BoolElement,
{
    fn fused_lstm_forward(
        input: burn::tensor::Tensor<Self, 3>,
        w_ih: burn::tensor::Tensor<Self, 2>,
        w_hh: burn::tensor::Tensor<Self, 2>,
        bias: Option<burn::tensor::Tensor<Self, 1>>,
        h0: Option<burn::tensor::Tensor<Self, 2>>,
        c0: Option<burn::tensor::Tensor<Self, 2>>,
    ) -> (burn::tensor::Tensor<Self, 3>, FusedLstmStateOut<Self>) {
        use burn::tensor::{Tensor, TensorPrimitive};

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

        // Precompute full-sequence input projection in one matmul + optional bias add.
        // Shape: [batch, seq_len, 4*hidden].
        let x_proj = input.matmul(w_ih.unsqueeze::<3>());
        let x_proj = match bias {
            Some(b) => x_proj + b.unsqueeze::<3>(),
            None => x_proj,
        };

        let mut outputs: Vec<Tensor<Self, 3>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Slice to [batch, 1, 4*hidden] and reshape to [batch, 4*hidden].
            let x_proj_t: Tensor<Self, 2> = x_proj
                .clone()
                .slice([0..batch, t..(t + 1), 0..four_hidden])
                .reshape([batch, four_hidden]);

            // Hidden projection for this step: h_prev @ w_hh  → [batch, 4*hidden].
            let h_proj: Tensor<Self, 2> = h.clone().matmul(w_hh.clone());

            // Drop to CubeTensor primitives for the kernel.
            let x_proj_t_prim: CubeTensor<R> = x_proj_t.into_primitive().tensor();
            let h_proj_prim: CubeTensor<R> = h_proj.into_primitive().tensor();
            let c_prim: CubeTensor<R> = c.into_primitive().tensor();

            let (h_next_prim, c_next_prim) =
                launch_cell::<R, F>(x_proj_t_prim, h_proj_prim, c_prim);

            h = Tensor::from_primitive(TensorPrimitive::Float(h_next_prim));
            c = Tensor::from_primitive(TensorPrimitive::Float(c_next_prim));

            outputs.push(h.clone().unsqueeze_dim(1));
        }

        let output = Tensor::cat(outputs, 1);
        (output, FusedLstmStateOut { hidden: h, cell: c })
    }

    fn fused_lstm_forward_train(
        input: burn::tensor::Tensor<Self, 3>,
        w_ih: burn::tensor::Tensor<Self, 2>,
        w_hh: burn::tensor::Tensor<Self, 2>,
        bias: burn::tensor::Tensor<Self, 1>,
    ) -> FusedLstmTrainOut<Self> {
        use burn::tensor::{Tensor, TensorPrimitive};

        let [batch, seq_len, _input_size] = input.dims();
        let [_, four_hidden] = w_ih.dims();
        assert!(
            four_hidden.is_multiple_of(4),
            "w_ih last dim must be divisible by 4 (got {four_hidden})"
        );
        let hidden = four_hidden / 4;
        let device = input.device();

        // Precompute full-sequence input projection in one matmul + bias add.
        // Shape: [batch, seq_len, 4*hidden].
        let x_proj = input.matmul(w_ih.unsqueeze::<3>()) + bias.unsqueeze::<3>();

        let mut h = Tensor::<Self, 2>::zeros([batch, hidden], &device);
        let mut c = Tensor::<Self, 2>::zeros([batch, hidden], &device);
        let mut h_steps: Vec<Tensor<Self, 3>> = Vec::with_capacity(seq_len);
        let mut c_steps: Vec<Tensor<Self, 3>> = Vec::with_capacity(seq_len);
        let mut z_steps: Vec<Tensor<Self, 3>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_proj_t: Tensor<Self, 2> = x_proj
                .clone()
                .slice([0..batch, t..(t + 1), 0..four_hidden])
                .reshape([batch, four_hidden]);

            // z = x_proj_t + h_prev @ w_hh: one matmul + one add. We save `z`
            // for BPTT and feed it to the cell kernel (which skips the add).
            // Net per-step GPU dispatches: 1 matmul + 1 add + 1 kernel = 3,
            // vs. ~13 through `default_fused_lstm_forward_train`.
            let h_proj: Tensor<Self, 2> = h.clone().matmul(w_hh.clone());
            let z: Tensor<Self, 2> = x_proj_t + h_proj;
            z_steps.push(z.clone().unsqueeze_dim(1));

            let z_prim: CubeTensor<R> = z.into_primitive().tensor();
            let c_prim: CubeTensor<R> = c.into_primitive().tensor();
            let (h_next_prim, c_next_prim) = launch_cell_train::<R, F>(z_prim, c_prim);

            h = Tensor::from_primitive(TensorPrimitive::Float(h_next_prim));
            c = Tensor::from_primitive(TensorPrimitive::Float(c_next_prim));

            h_steps.push(h.clone().unsqueeze_dim(1));
            c_steps.push(c.clone().unsqueeze_dim(1));
        }

        FusedLstmTrainOut {
            hidden_states: Tensor::cat(h_steps, 1),
            cell_states: Tensor::cat(c_steps, 1),
            pre_activations: Tensor::cat(z_steps, 1),
        }
    }
}
