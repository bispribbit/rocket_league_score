//! Backend dispatch for the fused LSTM, and the single-op BPTT backward pass.
//!
//! The reason this module exists is the autodiff graph. Expressed as plain
//! tensor ops, one LSTM layer records roughly `seq_len × 13` autograd nodes —
//! ~3900 at `seq_len = 300` — and backward then walks all of them one timestep
//! at a time. [`FusedLstmOp`] instead registers the whole sequence as a single
//! [`Backward`] op, so autodiff sees one edge from
//! `(input, w_ih, w_hh, bias) → output` and backward runs as a handful of
//! full-sequence matmuls. The graph becomes O(1) in sequence length rather than
//! O(`seq_len`).
//!
//! Dispatch is registered through [`backend_extension`], which generates the
//! glue that routes a call on the runtime-selected backend to the right impl:
//!
//! - **`Flex`** / **`Cube`** (every CubeCL runtime): [`forward_via_tensor_ops`],
//!   the plain-tensor cell loop from [`super::forward`].
//! - **`Autodiff<B, C>`**: the inner backend's training forward wrapped in one
//!   tracked [`FusedLstmOp`].
//!
//! The extension takes a required bias and no initial state — the shape
//! `SequenceModel` always calls it with. [`super::module::FusedLstm::forward`]
//! keeps every other configuration on the plain-tensor path, which stays
//! correct (just with the per-step graph).
//!
//! This module is not compiled for `wasm32`: the web build has only the Flex
//! CPU backend and runs inference without autodiff, so it routes straight to
//! [`super::forward`].

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::tensor::FloatTensor;
use burn::backend::{
    Autodiff, Backend, Cube, Dispatch, DispatchKindConversion, ExtensionType, Flex,
    backend_extension,
};
use burn::prelude::*;
use burn::tensor::DispatchTensor;
use burn::tensor::activation::{sigmoid, tanh};

use super::forward::{FusedLstmStateOut, fused_lstm_forward};

/// Output of [`FusedLstmBackend::fused_lstm_forward`].
#[derive(ExtensionType)]
pub struct FusedLstmForwardOut<B: Backend> {
    /// Hidden state at every step. `[batch, seq, hidden]`.
    pub output: FloatTensor<B>,
    /// Final hidden state. `[batch, hidden]`.
    pub hidden: FloatTensor<B>,
    /// Final cell state. `[batch, hidden]`.
    pub cell: FloatTensor<B>,
}

/// Output of [`FusedLstmBackend::fused_lstm_forward_train`]: the per-step
/// series BPTT needs, saved across the forward → backward boundary.
///
/// Memory footprint: `batch × seq_len × (6 * hidden) × 4` bytes. For
/// `batch=32`, `seq=300`, `hidden=256` that is ~60 MB per layer.
#[derive(ExtensionType)]
pub struct FusedLstmTrainOut<B: Backend> {
    /// Hidden states per step — also the module output. `[batch, seq, hidden]`.
    pub hidden_states: FloatTensor<B>,
    /// Post-update cell states per step. `[batch, seq, hidden]`.
    pub cell_states: FloatTensor<B>,
    /// Pre-activation gate stack, layout `[i | f | g | o]`. `[batch, seq, 4*hidden]`.
    pub pre_activations: FloatTensor<B>,
}

/// Per-backend fused LSTM forward, specialised so the autodiff backend can
/// record the whole sequence as one op.
#[backend_extension(Autodiff, Flex, Cube)]
pub trait FusedLstmBackend: Backend {
    /// Run the LSTM forward pass over a full sequence.
    ///
    /// - `input`: `[batch, seq_len, input_size]`
    /// - `w_ih`:  `[input_size,  4 * hidden]`, gate order `[i, f, g, o]`
    /// - `w_hh`:  `[hidden,      4 * hidden]`, gate order `[i, f, g, o]`
    /// - `bias`:  `[4 * hidden]`
    fn fused_lstm_forward(
        input: FloatTensor<Self>,
        w_ih: FloatTensor<Self>,
        w_hh: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FusedLstmForwardOut<Self>;

    /// Forward pass that also returns the per-step series BPTT needs.
    ///
    /// Only called on the backend *inside* `Autodiff`, by the tracked op.
    fn fused_lstm_forward_train(
        input: FloatTensor<Self>,
        w_ih: FloatTensor<Self>,
        w_hh: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FusedLstmTrainOut<Self>;
}

/// Runs the fused LSTM forward on whichever backend the tensors live on.
///
/// The bridge between the high-level [`Tensor`] the module holds and the
/// primitive-level extension trait: `into_dispatch` erases the tensor to the
/// runtime-selected backend, the generated `Dispatch` impl routes to the right
/// specialisation, and `from_dispatch` wraps the results back up.
pub(crate) fn dispatch_fused_lstm_forward(
    input: Tensor<3>,
    w_ih: Tensor<2>,
    w_hh: Tensor<2>,
    bias: Tensor<1>,
) -> (Tensor<3>, FusedLstmStateOut) {
    let FusedLstmForwardOut {
        output,
        hidden,
        cell,
    } = <Dispatch as FusedLstmBackend>::fused_lstm_forward(
        input.into_dispatch(),
        w_ih.into_dispatch(),
        w_hh.into_dispatch(),
        bias.into_dispatch(),
    );

    (
        Tensor::<3>::from_dispatch(output),
        FusedLstmStateOut {
            hidden: Tensor::<2>::from_dispatch(hidden),
            cell: Tensor::<2>::from_dispatch(cell),
        },
    )
}

/// Splits the `[i | f | g | o]` gate stack and applies the gate activations.
///
/// Shared by the forward recorder and the backward pass so the two can never
/// disagree about gate order.
struct GateActivations {
    input_gate: Tensor<3>,
    forget_gate: Tensor<3>,
    cell_candidate: Tensor<3>,
    output_gate: Tensor<3>,
}

fn gate_activations(
    pre_activations: &Tensor<3>,
    batch: usize,
    seq_len: usize,
    hidden: usize,
) -> GateActivations {
    let gate = |index: usize| {
        pre_activations.clone().slice([
            0..batch,
            0..seq_len,
            (index * hidden)..((index + 1) * hidden),
        ])
    };
    GateActivations {
        input_gate: sigmoid(gate(0)),
        forget_gate: sigmoid(gate(1)),
        cell_candidate: tanh(gate(2)),
        output_gate: sigmoid(gate(3)),
    }
}

/// Plain-tensor forward, for the backends that have no specialised path.
fn forward_via_tensor_ops<B: Backend>(
    input: FloatTensor<B>,
    w_ih: FloatTensor<B>,
    w_hh: FloatTensor<B>,
    bias: FloatTensor<B>,
) -> FusedLstmForwardOut<B>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let input = Tensor::<3>::from_primitive::<B>(input);
    let w_ih = Tensor::<2>::from_primitive::<B>(w_ih);
    let w_hh = Tensor::<2>::from_primitive::<B>(w_hh);
    let bias = Tensor::<1>::from_primitive::<B>(bias);

    let (output, FusedLstmStateOut { hidden, cell }) =
        fused_lstm_forward(input, w_ih, w_hh, Some(bias), None, None);

    FusedLstmForwardOut {
        output: into_primitive::<B, 3>(output),
        hidden: into_primitive::<B, 2>(hidden),
        cell: into_primitive::<B, 2>(cell),
    }
}

/// Plain-tensor forward that also records the per-step series for BPTT.
fn forward_train_via_tensor_ops<B: Backend>(
    input: FloatTensor<B>,
    w_ih: FloatTensor<B>,
    w_hh: FloatTensor<B>,
    bias: FloatTensor<B>,
) -> FusedLstmTrainOut<B>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let input = Tensor::<3>::from_primitive::<B>(input);
    let w_ih = Tensor::<2>::from_primitive::<B>(w_ih);
    let w_hh = Tensor::<2>::from_primitive::<B>(w_hh);
    let bias = Tensor::<1>::from_primitive::<B>(bias);

    let [batch, seq_len, _input_size] = input.dims();
    let [_, four_hidden] = w_ih.dims();
    assert!(
        four_hidden.is_multiple_of(4),
        "w_ih last dim must be divisible by 4 (got {four_hidden})"
    );
    let hidden = four_hidden / 4;
    let device = input.device();

    let x_projection = input.matmul(w_ih.unsqueeze::<3>()) + bias.unsqueeze::<3>();

    let mut hidden_state = Tensor::<2>::zeros([batch, hidden], &device);
    let mut cell_state = Tensor::<2>::zeros([batch, hidden], &device);
    let mut hidden_steps: Vec<Tensor<3>> = Vec::with_capacity(seq_len);
    let mut cell_steps: Vec<Tensor<3>> = Vec::with_capacity(seq_len);
    let mut pre_activation_steps: Vec<Tensor<3>> = Vec::with_capacity(seq_len);

    for step in 0..seq_len {
        let x_step: Tensor<2> = x_projection
            .clone()
            .slice([0..batch, step..(step + 1), 0..four_hidden])
            .reshape([batch, four_hidden]);

        let pre_activation = x_step + hidden_state.clone().matmul(w_hh.clone());
        pre_activation_steps.push(pre_activation.clone().unsqueeze_dim(1));

        let gate = |index: usize| {
            pre_activation
                .clone()
                .slice([0..batch, (index * hidden)..((index + 1) * hidden)])
        };
        let input_gate = sigmoid(gate(0));
        let forget_gate = sigmoid(gate(1));
        let cell_candidate = tanh(gate(2));
        let output_gate = sigmoid(gate(3));

        cell_state = forget_gate * cell_state + input_gate * cell_candidate;
        hidden_state = output_gate * tanh(cell_state.clone());

        hidden_steps.push(hidden_state.clone().unsqueeze_dim(1));
        cell_steps.push(cell_state.clone().unsqueeze_dim(1));
    }

    FusedLstmTrainOut {
        hidden_states: into_primitive::<B, 3>(Tensor::cat(hidden_steps, 1)),
        cell_states: into_primitive::<B, 3>(Tensor::cat(cell_steps, 1)),
        pre_activations: into_primitive::<B, 3>(Tensor::cat(pre_activation_steps, 1)),
    }
}

/// Unwraps a tensor back to `B`'s primitive.
///
/// Every call site here built the tensor from a `B` primitive moments earlier,
/// so the backend cannot have changed under it.
fn into_primitive<B: Backend, const D: usize>(tensor: Tensor<D>) -> FloatTensor<B>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    tensor
        .try_into_primitive::<B>()
        .expect("tensor was built from this backend's own primitive")
}

impl FusedLstmBackend for Flex {
    fn fused_lstm_forward(
        input: FloatTensor<Self>,
        w_ih: FloatTensor<Self>,
        w_hh: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FusedLstmForwardOut<Self> {
        forward_via_tensor_ops::<Self>(input, w_ih, w_hh, bias)
    }

    fn fused_lstm_forward_train(
        input: FloatTensor<Self>,
        w_ih: FloatTensor<Self>,
        w_hh: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FusedLstmTrainOut<Self> {
        forward_train_via_tensor_ops::<Self>(input, w_ih, w_hh, bias)
    }
}

impl FusedLstmBackend for Cube {
    fn fused_lstm_forward(
        input: FloatTensor<Self>,
        w_ih: FloatTensor<Self>,
        w_hh: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FusedLstmForwardOut<Self> {
        forward_via_tensor_ops::<Self>(input, w_ih, w_hh, bias)
    }

    fn fused_lstm_forward_train(
        input: FloatTensor<Self>,
        w_ih: FloatTensor<Self>,
        w_hh: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FusedLstmTrainOut<Self> {
        forward_train_via_tensor_ops::<Self>(input, w_ih, w_hh, bias)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Autodiff: the whole sequence as one tracked op
// ───────────────────────────────────────────────────────────────────────────

/// State saved across the forward → backward boundary.
///
/// All of it is recomputable from `(input, w_ih, w_hh, bias)`, but keeping the
/// pre-activation and cell series saves a full forward re-run in backward.
#[derive(Clone, Debug)]
struct LstmForwardState<B: Backend> {
    input: B::FloatTensorPrimitive,
    w_ih: B::FloatTensorPrimitive,
    w_hh: B::FloatTensorPrimitive,
    /// Pre-activation gate stack, chronological. `[batch, seq, 4*hidden]`.
    pre_activations: B::FloatTensorPrimitive,
    /// Cell states *after* each step, chronological. `[batch, seq, hidden]`.
    cell_states: B::FloatTensorPrimitive,
    /// Hidden states *after* each step (= forward output). `[batch, seq, hidden]`.
    hidden_states: B::FloatTensorPrimitive,
    batch: usize,
    seq_len: usize,
    hidden: usize,
    input_size: usize,
}

/// Zero-sized marker registering the LSTM backward pass.
#[derive(Debug)]
struct FusedLstmOp;

impl<B: FusedLstmBackend> Backward<B, 4> for FusedLstmOp
where
    DispatchTensor: DispatchKindConversion<B>,
{
    type State = LstmForwardState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 4>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let LstmForwardState {
            input,
            w_ih,
            w_hh,
            pre_activations,
            cell_states,
            hidden_states,
            batch,
            seq_len,
            hidden,
            input_size,
        } = ops.state;
        let four_hidden = 4 * hidden;

        let grad_output = Tensor::<3>::from_primitive::<B>(grads.consume::<B>(&ops.node));
        let input = Tensor::<3>::from_primitive::<B>(input);
        let w_ih = Tensor::<2>::from_primitive::<B>(w_ih);
        let w_hh = Tensor::<2>::from_primitive::<B>(w_hh);
        let pre_activations = Tensor::<3>::from_primitive::<B>(pre_activations);
        let cell_states = Tensor::<3>::from_primitive::<B>(cell_states);
        let hidden_states = Tensor::<3>::from_primitive::<B>(hidden_states);

        let device = grad_output.device();

        // Recompute gate activations from the saved pre-activations, batched
        // over the whole sequence rather than per step.
        let GateActivations {
            input_gate,
            forget_gate,
            cell_candidate,
            output_gate,
        } = gate_activations(&pre_activations, batch, seq_len, hidden);
        let cell_tanh = tanh(cell_states.clone());

        let mut grad_hidden_next = Tensor::<2>::zeros([batch, hidden], &device);
        let mut grad_cell_next = Tensor::<2>::zeros([batch, hidden], &device);
        let mut grad_pre_activation_steps: Vec<Tensor<3>> = Vec::with_capacity(seq_len);

        for step in (0..seq_len).rev() {
            let at_step = |series: &Tensor<3>| -> Tensor<2> {
                series
                    .clone()
                    .slice([0..batch, step..(step + 1), 0..hidden])
                    .reshape([batch, hidden])
            };

            let input_gate_step = at_step(&input_gate);
            let forget_gate_step = at_step(&forget_gate);
            let cell_candidate_step = at_step(&cell_candidate);
            let output_gate_step = at_step(&output_gate);
            let cell_tanh_step = at_step(&cell_tanh);

            let cell_previous = if step == 0 {
                Tensor::<2>::zeros([batch, hidden], &device)
            } else {
                cell_states
                    .clone()
                    .slice([0..batch, (step - 1)..step, 0..hidden])
                    .reshape([batch, hidden])
            };

            let ones = Tensor::<2>::ones([batch, hidden], &device);

            // h = o * tanh(c)  ->  do = dh * tanh(c);  dc += dh * o * (1 - tanh(c)^2)
            let grad_hidden = at_step(&grad_output) + grad_hidden_next.clone();
            let grad_output_gate = grad_hidden.clone() * cell_tanh_step.clone();
            let grad_cell = grad_hidden
                * output_gate_step.clone()
                * (ones.clone() - cell_tanh_step.clone() * cell_tanh_step)
                + grad_cell_next;

            // c = f * c_prev + i * g
            let grad_forget_gate = grad_cell.clone() * cell_previous;
            let grad_cell_previous = grad_cell.clone() * forget_gate_step.clone();
            let grad_input_gate = grad_cell.clone() * cell_candidate_step.clone();
            let grad_cell_candidate = grad_cell * input_gate_step.clone();

            // Activation backward: sigmoid'(x) = s(1 - s), tanh'(x) = 1 - tanh^2
            let grad_input_pre =
                grad_input_gate * input_gate_step.clone() * (ones.clone() - input_gate_step);
            let grad_forget_pre =
                grad_forget_gate * forget_gate_step.clone() * (ones.clone() - forget_gate_step);
            let grad_candidate_pre = grad_cell_candidate
                * (ones.clone() - cell_candidate_step.clone() * cell_candidate_step);
            let grad_output_pre =
                grad_output_gate * output_gate_step.clone() * (ones - output_gate_step);

            // Gate order along the last dim: [i, f, g, o], matching forward.
            let grad_pre_activation: Tensor<2> = Tensor::cat(
                vec![
                    grad_input_pre,
                    grad_forget_pre,
                    grad_candidate_pre,
                    grad_output_pre,
                ],
                1,
            );

            // Recurrent edge into the previous step.
            grad_hidden_next = grad_pre_activation.clone().matmul(w_hh.clone().transpose());
            grad_cell_next = grad_cell_previous;

            grad_pre_activation_steps.push(grad_pre_activation.unsqueeze_dim(1));
        }

        grad_pre_activation_steps.reverse();
        let grad_x_projection: Tensor<3> = Tensor::cat(grad_pre_activation_steps, 1);

        // x_projection = input @ w_ih + bias
        let grad_input: Tensor<3> = grad_x_projection
            .clone()
            .matmul(w_ih.transpose().unsqueeze::<3>());

        let input_flat: Tensor<2> = input.reshape([batch * seq_len, input_size]);
        let grad_x_projection_flat: Tensor<2> =
            grad_x_projection.reshape([batch * seq_len, four_hidden]);
        let grad_w_ih: Tensor<2> = input_flat
            .transpose()
            .matmul(grad_x_projection_flat.clone());
        let grad_bias: Tensor<1> = grad_x_projection_flat
            .clone()
            .sum_dim(0)
            .reshape([four_hidden]);

        // grad_w_hh = sum over steps of h_{t-1}^T @ dz_t. Build h_{t-1} by
        // prepending a zero step and dropping the last hidden state.
        let hidden_previous: Tensor<3> = if seq_len > 1 {
            let zero_step: Tensor<3> = Tensor::zeros([batch, 1, hidden], &device);
            let shifted: Tensor<3> = hidden_states.slice([0..batch, 0..(seq_len - 1), 0..hidden]);
            Tensor::cat(vec![zero_step, shifted], 1)
        } else {
            Tensor::zeros([batch, 1, hidden], &device)
        };
        let hidden_previous_flat: Tensor<2> = hidden_previous.reshape([batch * seq_len, hidden]);
        let grad_w_hh: Tensor<2> = hidden_previous_flat
            .transpose()
            .matmul(grad_x_projection_flat);

        let [parent_input, parent_w_ih, parent_w_hh, parent_bias] = ops.parents;
        if let Some(node) = parent_input {
            grads.register::<B>(node.id, into_primitive::<B, 3>(grad_input));
        }
        if let Some(node) = parent_w_ih {
            grads.register::<B>(node.id, into_primitive::<B, 2>(grad_w_ih));
        }
        if let Some(node) = parent_w_hh {
            grads.register::<B>(node.id, into_primitive::<B, 2>(grad_w_hh));
        }
        if let Some(node) = parent_bias {
            grads.register::<B>(node.id, into_primitive::<B, 1>(grad_bias));
        }
    }
}

impl<B, C> FusedLstmBackend for Autodiff<B, C>
where
    B: FusedLstmBackend,
    C: CheckpointStrategy,
    DispatchTensor: DispatchKindConversion<B>,
    DispatchTensor: DispatchKindConversion<Self>,
{
    fn fused_lstm_forward(
        input: FloatTensor<Self>,
        w_ih: FloatTensor<Self>,
        w_hh: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FusedLstmForwardOut<Self> {
        let batch_and_shape = {
            let input_tensor = Tensor::<3>::from_primitive::<Self>(input.clone());
            let w_ih_tensor = Tensor::<2>::from_primitive::<Self>(w_ih.clone());
            let [batch, seq_len, input_size] = input_tensor.dims();
            let [_, four_hidden] = w_ih_tensor.dims();
            assert!(
                four_hidden.is_multiple_of(4),
                "w_ih last dim must be divisible by 4 (got {four_hidden})"
            );
            (batch, seq_len, input_size, four_hidden / 4)
        };
        let (batch, seq_len, input_size, hidden) = batch_and_shape;

        // Take a guard on each parent node before consuming its primitive.
        let input_node = input.node();
        let w_ih_node = w_ih.node();
        let w_hh_node = w_hh.node();
        let bias_node = bias.node();

        let (input_primitive, _) = input.into_parts();
        let (w_ih_primitive, _) = w_ih.into_parts();
        let (w_hh_primitive, _) = w_hh.into_parts();
        let (bias_primitive, _) = bias.into_parts();

        // The inner backend runs the real LSTM math and hands back the series
        // backward needs.
        let FusedLstmTrainOut {
            hidden_states,
            cell_states,
            pre_activations,
        } = B::fused_lstm_forward_train(
            input_primitive.clone(),
            w_ih_primitive.clone(),
            w_hh_primitive.clone(),
            bias_primitive,
        );

        let saved = LstmForwardState::<B> {
            input: input_primitive,
            w_ih: w_ih_primitive,
            w_hh: w_hh_primitive,
            pre_activations,
            cell_states,
            hidden_states: hidden_states.clone(),
            batch,
            seq_len,
            hidden,
            input_size,
        };

        let tracked_output = match FusedLstmOp
            .prepare::<C>([input_node, w_ih_node, w_hh_node, bias_node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(saved, hidden_states),
            OpsKind::UnTracked(prep) => prep.finish(hidden_states),
        };

        let output = Tensor::<3>::from_primitive::<Self>(tracked_output);
        let device = output.device();

        // Final hidden is the last step of the tracked output, so it stays on
        // the graph. Final cell is a detached zeros leaf: `SequenceModel` never
        // reads it, and promoting this to a multi-output op would only be worth
        // it if that changed.
        let final_hidden: Tensor<2> = output
            .clone()
            .slice([0..batch, (seq_len - 1)..seq_len, 0..hidden])
            .reshape([batch, hidden]);
        let final_cell: Tensor<2> = Tensor::zeros([batch, hidden], &device);

        FusedLstmForwardOut {
            output: into_primitive::<Self, 3>(output),
            hidden: into_primitive::<Self, 2>(final_hidden),
            cell: into_primitive::<Self, 2>(final_cell),
        }
    }

    fn fused_lstm_forward_train(
        input: FloatTensor<Self>,
        w_ih: FloatTensor<Self>,
        w_hh: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FusedLstmTrainOut<Self> {
        // Reached only if something asks an autodiff backend for the training
        // series directly; the tracked op always calls the inner backend.
        forward_train_via_tensor_ops::<Self>(input, w_ih, w_hh, bias)
    }
}
