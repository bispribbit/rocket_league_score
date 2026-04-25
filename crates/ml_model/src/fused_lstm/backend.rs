//! Backend dispatch trait for the fused LSTM forward pass.
//!
//! All LSTM computation in this workspace goes through [`FusedLstmBackend`]:
//!
//! - **`NdArray`**: [`default_fused_lstm_forward`] / [`default_fused_lstm_forward_train`]
//!   — generic `Tensor`-op loop, used for web inference (WASM/CPU).
//! - **`CubeBackend<R, …>`** (non-WASM): one CubeCL kernel per
//!   timestep for the Wgpu training backend (see `cube_kernel.rs`).
//! - **`Autodiff<B, C>`**: wraps the inner backend's training forward in a
//!   single `Backward<B, 4>` op so the autograd graph is O(1) in sequence length.
//!
//! If you add a new backend, add a `impl FusedLstmBackend for MyBackend` that
//! routes to [`default_fused_lstm_forward`] — that's all that's required.

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::ndarray::{
    FloatNdArrayElement, IntNdArrayElement, NdArrayTensor, QuantElement, SharedArray,
};
use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::TensorPrimitive;
use burn::tensor::activation::{sigmoid, tanh};

/// Final hidden / cell state produced by a fused LSTM forward pass.
#[derive(Debug, Clone)]
pub struct FusedLstmStateOut<B: Backend> {
    /// Final hidden state. Shape: `[batch, hidden]`.
    pub hidden: Tensor<B, 2>,
    /// Final cell state. Shape: `[batch, hidden]`.
    pub cell: Tensor<B, 2>,
}

/// Intermediate state produced by [`FusedLstmBackend::fused_lstm_forward_train`]
/// and saved across the forward → backward boundary for BPTT.
///
/// Memory footprint: `batch × seq_len × (6 * hidden) × sizeof(F)`. For
/// `batch=32`, `seq=300`, `hidden=256`, f32 that's ~60 MB per layer.
#[derive(Debug, Clone)]
pub struct FusedLstmTrainOut<B: Backend> {
    /// Hidden states per step — also the module output. `[batch, seq, hidden]`.
    pub hidden_states: Tensor<B, 3>,
    /// Post-update cell states per step. `[batch, seq, hidden]`.
    pub cell_states: Tensor<B, 3>,
    /// Pre-activation gate stack, layout `[i | f | g | o]`. `[batch, seq, 4*hidden]`.
    pub pre_activations: Tensor<B, 3>,
}

/// Extension trait implemented per backend so the forward pass can be
/// specialised (CubeCL kernel on `CubeBackend`, generic `Tensor` ops
/// elsewhere, single Backward-op wrapper on `Autodiff<B>`).
pub trait FusedLstmBackend: Backend {
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
    /// - `state`:  final `(hidden, cell)`.
    fn fused_lstm_forward(
        input: Tensor<Self, 3>,
        w_ih: Tensor<Self, 2>,
        w_hh: Tensor<Self, 2>,
        bias: Option<Tensor<Self, 1>>,
        h0: Option<Tensor<Self, 2>>,
        c0: Option<Tensor<Self, 2>>,
    ) -> (Tensor<Self, 3>, FusedLstmStateOut<Self>);

    /// Training-aware forward that also returns the intermediate state needed
    /// by the single-op BPTT `Backward` implementation in `Autodiff<Self>`.
    ///
    /// This is the hot path actually exercised during training: the generic
    /// `Autodiff<B>` specialisation unwraps its primitives and calls
    /// `B::fused_lstm_forward_train(…)` on the inner backend.
    ///
    /// Always uses zero initial `h0` / `c0` and requires a bias — matching how
    /// [`FusedLstm`](super::module::FusedLstm) is configured in
    /// `SequenceModel`. Backends that specialise this method (`CubeBackend`)
    /// dispatch a single GPU kernel per timestep; the default routes through
    /// standard `Tensor` ops.
    fn fused_lstm_forward_train(
        input: Tensor<Self, 3>,
        w_ih: Tensor<Self, 2>,
        w_hh: Tensor<Self, 2>,
        bias: Tensor<Self, 1>,
    ) -> FusedLstmTrainOut<Self> {
        default_fused_lstm_forward_train::<Self>(input, w_ih, w_hh, bias)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Reference implementation
// ───────────────────────────────────────────────────────────────────────────

/// Reference forward pass. High-level burn tensor ops only — correct on
/// every backend, but performs ~1 big matmul + ~10 small ops per timestep.
///
/// Beats [`burn::nn::Lstm`] on kernel count by:
/// 1. Precomputing the full-sequence input projection `input @ W_ih + b` in
///    one matmul outside the timestep loop (vs. 4 per step in burn-nn).
/// 2. Performing `h_{t-1} @ W_hh` as a single batched matmul rather than
///    four per-gate linear transforms.
pub fn default_fused_lstm_forward<B: Backend>(
    input: Tensor<B, 3>,
    w_ih: Tensor<B, 2>,
    w_hh: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
    h0: Option<Tensor<B, 2>>,
    c0: Option<Tensor<B, 2>>,
) -> (Tensor<B, 3>, FusedLstmStateOut<B>) {
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

    let mut outputs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let x_t: Tensor<B, 2> = x_proj
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

    let outputs_expanded: Vec<Tensor<B, 3>> =
        outputs.into_iter().map(|t| t.unsqueeze_dim(1)).collect();
    let output = Tensor::cat(outputs_expanded, 1);

    (output, FusedLstmStateOut { hidden: h, cell: c })
}

/// Reference training forward. Same cell math as [`default_fused_lstm_forward`]
/// but records per-step pre-activations, cell states, and hidden states so
/// BPTT can run as batched full-sequence ops in backward.
///
/// Used as the fallback path for backends that don't specialise
/// [`FusedLstmBackend::fused_lstm_forward_train`].
pub fn default_fused_lstm_forward_train<B: Backend>(
    input: Tensor<B, 3>,
    w_ih: Tensor<B, 2>,
    w_hh: Tensor<B, 2>,
    bias: Tensor<B, 1>,
) -> FusedLstmTrainOut<B> {
    let [batch, seq_len, _input_size] = input.dims();
    let [_, four_hidden] = w_ih.dims();
    assert!(
        four_hidden.is_multiple_of(4),
        "w_ih last dim must be divisible by 4 (got {four_hidden})"
    );
    let hidden = four_hidden / 4;
    let device = input.device();

    let x_proj = input.matmul(w_ih.unsqueeze::<3>()) + bias.unsqueeze::<3>();

    let mut h = Tensor::<B, 2>::zeros([batch, hidden], &device);
    let mut c = Tensor::<B, 2>::zeros([batch, hidden], &device);
    let mut h_steps: Vec<Tensor<B, 3>> = Vec::with_capacity(seq_len);
    let mut c_steps: Vec<Tensor<B, 3>> = Vec::with_capacity(seq_len);
    let mut z_steps: Vec<Tensor<B, 3>> = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let x_t: Tensor<B, 2> = x_proj
            .clone()
            .slice([0..batch, t..(t + 1), 0..four_hidden])
            .reshape([batch, four_hidden]);

        let z = x_t + h.clone().matmul(w_hh.clone());
        z_steps.push(z.clone().unsqueeze_dim(1));

        let i_gate = sigmoid(z.clone().slice([0..batch, 0..hidden]));
        let f_gate = sigmoid(z.clone().slice([0..batch, hidden..2 * hidden]));
        let g_gate = tanh(z.clone().slice([0..batch, 2 * hidden..3 * hidden]));
        let o_gate = sigmoid(z.slice([0..batch, 3 * hidden..4 * hidden]));

        c = f_gate * c + i_gate * g_gate;
        h = o_gate * tanh(c.clone());

        h_steps.push(h.clone().unsqueeze_dim(1));
        c_steps.push(c.clone().unsqueeze_dim(1));
    }

    FusedLstmTrainOut {
        hidden_states: Tensor::cat(h_steps, 1),
        cell_states: Tensor::cat(c_steps, 1),
        pre_activations: Tensor::cat(z_steps, 1),
    }
}

// ───────────────────────────────────────────────────────────────────────────
// NdArray impl (web inference + CPU tests)
// ───────────────────────────────────────────────────────────────────────────

impl<E, I, Q> FusedLstmBackend for NdArray<E, I, Q>
where
    E: FloatNdArrayElement,
    I: IntNdArrayElement,
    Q: QuantElement,
    NdArrayTensor: From<SharedArray<E>> + From<SharedArray<I>>,
{
    fn fused_lstm_forward(
        input: Tensor<Self, 3>,
        w_ih: Tensor<Self, 2>,
        w_hh: Tensor<Self, 2>,
        bias: Option<Tensor<Self, 1>>,
        h0: Option<Tensor<Self, 2>>,
        c0: Option<Tensor<Self, 2>>,
    ) -> (Tensor<Self, 3>, FusedLstmStateOut<Self>) {
        default_fused_lstm_forward::<Self>(input, w_ih, w_hh, bias, h0, c0)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Autodiff impl — single BPTT Backward op
// ───────────────────────────────────────────────────────────────────────────
//
// The entire LSTM forward is recorded as one `Backward<B,4>` op.
// Autodiff only sees one edge from (input, w_ih, w_hh, bias) → output.
// Backward executes as a handful of full-sequence matmuls, collapsing the
// ~3900 autograd nodes that per-step tracking would produce (300 steps ×
// ~13 ops) into ~10.

/// State saved across the forward → backward boundary.
///
/// Everything needed by BPTT is recomputable from `(input, w_ih, w_hh, bias)`
/// alone, but keeping the pre-activation `z` and cell-state `c` series cached
/// saves one full forward re-run during backward. Memory: `(batch × seq ×
/// (input + 2 * hidden + 4 * hidden + hidden)) × 4 bytes` — for `batch=32`,
/// `seq=300`, `hidden=256`, `input=128` that's ~12 MB per layer on GPU.
#[derive(Clone, Debug)]
struct LstmFwdState<B: Backend> {
    input: B::FloatTensorPrimitive,
    w_ih: B::FloatTensorPrimitive,
    w_hh: B::FloatTensorPrimitive,
    /// Pre-activation gate stack, chronological: `z[:, t, :]` shape `[batch, 4*hidden]`.
    z_all: B::FloatTensorPrimitive,
    /// Cell states *after* each step, chronological: `c[:, t, :]` shape `[batch, hidden]`.
    c_all: B::FloatTensorPrimitive,
    /// Hidden states *after* each step (= forward output): `h[:, t, :]` shape `[batch, hidden]`.
    h_all: B::FloatTensorPrimitive,
    batch: usize,
    seq_len: usize,
    hidden: usize,
    input_size: usize,
}

/// Zero-sized marker for the LSTM Backward op registration.
#[derive(Debug)]
struct FusedLstmOp;

impl<B: FusedLstmBackend> Backward<B, 4> for FusedLstmOp {
    type State = LstmFwdState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 4>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let st = ops.state;
        let grad_out_prim = grads.consume::<B>(&ops.node);

        let LstmFwdState {
            input,
            w_ih,
            w_hh,
            z_all,
            c_all,
            h_all,
            batch,
            seq_len,
            hidden,
            input_size,
        } = st;
        let four_hidden = 4 * hidden;

        let grad_out: Tensor<B, 3> = Tensor::from_primitive(TensorPrimitive::Float(grad_out_prim));
        let input_t: Tensor<B, 3> = Tensor::from_primitive(TensorPrimitive::Float(input));
        let w_ih_t: Tensor<B, 2> = Tensor::from_primitive(TensorPrimitive::Float(w_ih));
        let w_hh_t: Tensor<B, 2> = Tensor::from_primitive(TensorPrimitive::Float(w_hh));
        let z_all_t: Tensor<B, 3> = Tensor::from_primitive(TensorPrimitive::Float(z_all));
        let c_all_t: Tensor<B, 3> = Tensor::from_primitive(TensorPrimitive::Float(c_all));
        let h_all_t: Tensor<B, 3> = Tensor::from_primitive(TensorPrimitive::Float(h_all));

        let device = grad_out.device();

        // Recompute gate activations from the saved pre-activations, batched over the full seq.
        let i_all = sigmoid(z_all_t.clone().slice([0..batch, 0..seq_len, 0..hidden]));
        let f_all = sigmoid(
            z_all_t
                .clone()
                .slice([0..batch, 0..seq_len, hidden..2 * hidden]),
        );
        let g_all = tanh(
            z_all_t
                .clone()
                .slice([0..batch, 0..seq_len, 2 * hidden..3 * hidden]),
        );
        let o_all = sigmoid(z_all_t.slice([0..batch, 0..seq_len, 3 * hidden..4 * hidden]));
        let tanh_c_all = tanh(c_all_t.clone());

        // BPTT loop. Collects per-step dz to build `dx_proj_all` as one tensor afterwards.
        let mut dh_next: Tensor<B, 2> = Tensor::zeros([batch, hidden], &device);
        let mut dc_next: Tensor<B, 2> = Tensor::zeros([batch, hidden], &device);
        let mut dz_steps: Vec<Tensor<B, 3>> = Vec::with_capacity(seq_len);

        for t in (0..seq_len).rev() {
            let slice_t = |t_big: &Tensor<B, 3>| -> Tensor<B, 2> {
                t_big
                    .clone()
                    .slice([0..batch, t..(t + 1), 0..hidden])
                    .reshape([batch, hidden])
            };

            let i_t = slice_t(&i_all);
            let f_t = slice_t(&f_all);
            let g_t = slice_t(&g_all);
            let o_t = slice_t(&o_all);
            let tc_t = slice_t(&tanh_c_all);

            let c_prev = if t == 0 {
                Tensor::<B, 2>::zeros([batch, hidden], &device)
            } else {
                c_all_t
                    .clone()
                    .slice([0..batch, (t - 1)..t, 0..hidden])
                    .reshape([batch, hidden])
            };

            // h = o * tanh(c)  →  do = dh * tanh(c);  dc += dh * o * (1 - tanh(c)^2)
            let dh_t = slice_t(&grad_out);
            let dh = dh_t + dh_next.clone();
            let do_t = dh.clone() * tc_t.clone();
            let ones_hidden = Tensor::<B, 2>::ones([batch, hidden], &device);
            let dc = dh * o_t.clone() * (ones_hidden.clone() - tc_t.clone() * tc_t) + dc_next;

            // c = f*c_prev + i*g  →  gate gradients + dc_prev
            let df_t = dc.clone() * c_prev;
            let dc_prev = dc.clone() * f_t.clone();
            let di_t = dc.clone() * g_t.clone();
            let dg_t = dc * i_t.clone();

            // Activation backward: sigmoid'(x)=σ(1-σ), tanh'(x)=1-tanh²
            let di_pre = di_t * i_t.clone() * (ones_hidden.clone() - i_t);
            let df_pre = df_t * f_t.clone() * (ones_hidden.clone() - f_t);
            let dg_pre = dg_t * (ones_hidden - g_t.clone() * g_t);
            let do_pre =
                do_t * o_t.clone() * (Tensor::<B, 2>::ones([batch, hidden], &device) - o_t);

            // Gate order along last dim: [i, f, g, o], matching forward.
            let dz_t: Tensor<B, 2> = Tensor::cat(vec![di_pre, df_pre, dg_pre, do_pre], 1);

            // Contribution to next iteration via recurrent edge: h_{t-1} ← dz_t @ W_hh^T
            dh_next = dz_t.clone().matmul(w_hh_t.clone().transpose());
            dc_next = dc_prev;

            dz_steps.push(dz_t.unsqueeze_dim(1));
        }

        dz_steps.reverse();
        let dx_proj_all: Tensor<B, 3> = Tensor::cat(dz_steps, 1);

        // x_proj = input @ W_ih + bias
        //   d_input = dx_proj @ W_ih^T
        //   d_w_ih = input^T @ dx_proj  (collapsed over batch × seq)
        //   d_bias = sum over (batch, seq) of dx_proj
        let d_input: Tensor<B, 3> = dx_proj_all
            .clone()
            .matmul(w_ih_t.transpose().unsqueeze::<3>());

        let input_2d: Tensor<B, 2> = input_t.reshape([batch * seq_len, input_size]);
        let dx_proj_2d: Tensor<B, 2> = dx_proj_all.reshape([batch * seq_len, four_hidden]);
        let d_w_ih: Tensor<B, 2> = input_2d.transpose().matmul(dx_proj_2d.clone());

        let d_bias: Tensor<B, 1> = dx_proj_2d.clone().sum_dim(0).reshape([four_hidden]);

        // d_w_hh = sum_t h_{t-1}^T @ dz_t
        // Build h_prev_all by prepending a zeros "step 0 prev" and dropping the last h.
        let h_prev_all: Tensor<B, 3> = if seq_len > 1 {
            let zeros_first: Tensor<B, 3> = Tensor::zeros([batch, 1, hidden], &device);
            let h_past: Tensor<B, 3> = h_all_t.slice([0..batch, 0..(seq_len - 1), 0..hidden]);
            Tensor::cat(vec![zeros_first, h_past], 1)
        } else {
            Tensor::zeros([batch, 1, hidden], &device)
        };
        let h_prev_2d: Tensor<B, 2> = h_prev_all.reshape([batch * seq_len, hidden]);
        let d_w_hh: Tensor<B, 2> = h_prev_2d.transpose().matmul(dx_proj_2d);

        // Register gradients on whichever parents required them.
        let [p_input, p_w_ih, p_w_hh, p_bias] = ops.parents;
        if let Some(node) = p_input {
            grads.register::<B>(node.id, d_input.into_primitive().tensor());
        }
        if let Some(node) = p_w_ih {
            grads.register::<B>(node.id, d_w_ih.into_primitive().tensor());
        }
        if let Some(node) = p_w_hh {
            grads.register::<B>(node.id, d_w_hh.into_primitive().tensor());
        }
        if let Some(node) = p_bias {
            grads.register::<B>(node.id, d_bias.into_primitive().tensor());
        }
    }
}

/// `Autodiff<B>` specialisation: wraps the inner backend's fused LSTM (or
/// the default path) in a single tracked op so autodiff only sees one
/// edge from `(input, w_ih, w_hh, bias)` → `output`.
///
/// The fast path requires `bias = Some` and no initial state — matches how
/// [`FusedLstm`](super::module::FusedLstm) is used in `SequenceModel`. Any
/// other configuration falls through to [`default_fused_lstm_forward`] so
/// unusual call sites stay correct (just slower).
impl<B, C> FusedLstmBackend for Autodiff<B, C>
where
    B: FusedLstmBackend,
    C: CheckpointStrategy,
{
    fn fused_lstm_forward(
        input: Tensor<Self, 3>,
        w_ih: Tensor<Self, 2>,
        w_hh: Tensor<Self, 2>,
        bias: Option<Tensor<Self, 1>>,
        h0: Option<Tensor<Self, 2>>,
        c0: Option<Tensor<Self, 2>>,
    ) -> (Tensor<Self, 3>, FusedLstmStateOut<Self>) {
        // Slow-path guard: uncommon configurations fall back to the per-step
        // tracked implementation. Our `FusedLstm` module never hits these in
        // `SequenceModel`.
        let Some(bias) = bias else {
            return default_fused_lstm_forward::<Self>(input, w_ih, w_hh, None, h0, c0);
        };
        if h0.is_some() || c0.is_some() {
            return default_fused_lstm_forward::<Self>(input, w_ih, w_hh, Some(bias), h0, c0);
        }

        let [batch, seq_len, input_size] = input.dims();
        let [_, four_hidden] = w_ih.dims();
        assert!(
            four_hidden.is_multiple_of(4),
            "w_ih last dim must be divisible by 4 (got {four_hidden})"
        );
        let hidden = four_hidden / 4;

        // Unwrap autodiff tensors → (primitive, node) pairs.
        let input_ad = input.into_primitive().tensor();
        let w_ih_ad = w_ih.into_primitive().tensor();
        let w_hh_ad = w_hh.into_primitive().tensor();
        let bias_ad = bias.into_primitive().tensor();
        let input_node = input_ad.node.clone();
        let w_ih_node = w_ih_ad.node.clone();
        let w_hh_node = w_hh_ad.node.clone();
        let bias_node = bias_ad.node.clone();

        // Wrap inner primitives as inner-backend tensors for the forward.
        let input_primitive = input_ad.primitive;
        let w_ih_primitive = w_ih_ad.primitive;
        let w_hh_primitive = w_hh_ad.primitive;
        let bias_primitive = bias_ad.primitive;

        let input_b: Tensor<B, 3> =
            Tensor::from_primitive(TensorPrimitive::Float(input_primitive.clone()));
        let w_ih_b: Tensor<B, 2> =
            Tensor::from_primitive(TensorPrimitive::Float(w_ih_primitive.clone()));
        let w_hh_b: Tensor<B, 2> =
            Tensor::from_primitive(TensorPrimitive::Float(w_hh_primitive.clone()));
        let bias_b: Tensor<B, 1> = Tensor::from_primitive(TensorPrimitive::Float(bias_primitive));

        // Delegate the actual LSTM math to the inner backend's specialised
        // training forward — CubeCL kernel on `CubeBackend`, reference Tensor
        // ops otherwise. This is what collapses the ~13-dispatch-per-step
        // pattern into a single cell-kernel launch on GPU.
        let FusedLstmTrainOut {
            hidden_states,
            cell_states,
            pre_activations,
        } = B::fused_lstm_forward_train(input_b, w_ih_b, w_hh_b, bias_b);
        let output_device = hidden_states.device();

        let saved = LstmFwdState::<B> {
            input: input_primitive,
            w_ih: w_ih_primitive,
            w_hh: w_hh_primitive,
            z_all: pre_activations.into_primitive().tensor(),
            c_all: cell_states.into_primitive().tensor(),
            h_all: hidden_states.clone().into_primitive().tensor(),
            batch,
            seq_len,
            hidden,
            input_size,
        };

        let output_prim = hidden_states.into_primitive().tensor();

        let output_ad = match FusedLstmOp
            .prepare::<C>([input_node, w_ih_node, w_hh_node, bias_node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(saved, output_prim),
            OpsKind::UnTracked(prep) => prep.finish(output_prim),
        };

        let output: Tensor<Self, 3> = Tensor::from_primitive(TensorPrimitive::Float(output_ad));
        // Final hidden is the last timestep of the tracked output; slice stays
        // on the autograd graph so downstream uses flow gradients correctly.
        // Final cell is returned as a detached zeros leaf — `SequenceModel`
        // never uses it; promote to a multi-output op only if that changes.
        let final_hidden: Tensor<Self, 2> = output
            .clone()
            .slice([0..batch, (seq_len - 1)..seq_len, 0..hidden])
            .reshape([batch, hidden]);
        let final_cell: Tensor<Self, 2> = Tensor::zeros([batch, hidden], &output_device);

        (
            output,
            FusedLstmStateOut {
                hidden: final_hidden,
                cell: final_cell,
            },
        )
    }
}
