//! [`FusedLstm`] module. Drop-in replacement for [`burn::nn::Lstm`] using
//! a concatenated 4-gate weight layout.

use burn::config::Config;
use burn::module::{Initializer, Module, Param};
use burn::prelude::*;

use super::forward::{FusedLstmStateOut, fused_lstm_forward};

/// Initial (or returned final) state for a [`FusedLstm`] forward pass.
///
/// Mirrors [`burn::nn::LstmState`] layout so the two modules stay
/// interchangeable at call sites.
#[derive(Debug, Clone)]
pub struct FusedLstmState<const D: usize> {
    /// Cell state.
    pub cell: Tensor<D>,
    /// Hidden state.
    pub hidden: Tensor<D>,
}

impl<const D: usize> FusedLstmState<D> {
    /// Build a new state.
    pub const fn new(cell: Tensor<D>, hidden: Tensor<D>) -> Self {
        Self { cell, hidden }
    }
}

/// Configuration for [`FusedLstm`].
#[derive(Config, Debug)]
pub struct FusedLstmConfig {
    /// Input feature size.
    pub d_input: usize,
    /// Hidden state size.
    pub d_hidden: usize,
    /// Whether to apply a bias to the gate projections.
    pub bias: bool,
    /// Weight initializer. Matches `burn::nn::LstmConfig`'s default.
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub initializer: Initializer,
    /// Initialise the forget-gate bias slice to `1.0` (zeros elsewhere) when
    /// `bias = true`.
    ///
    /// Jozefowicz et al. (2015) and standard practice in PyTorch /
    /// TensorFlow / JAX: a forget-gate bias of 1 keeps the cell state
    /// flowing at initialisation (`sigmoid(0 + 1) = 0.73` vs. `sigmoid(0) = 0.5`)
    /// and dramatically shortens the early-training phase where the LSTM has
    /// to learn to remember anything.
    #[config(default = "true")]
    pub forget_gate_bias_one: bool,
}

impl FusedLstmConfig {
    /// Initialise a new [`FusedLstm`] module on the given device.
    pub fn init(&self, device: &Device) -> FusedLstm {
        let four_hidden = 4 * self.d_hidden;

        let w_ih = self.initializer.init_with(
            [self.d_input, four_hidden],
            Some(self.d_input),
            Some(four_hidden),
            device,
        );
        let w_hh = self.initializer.init_with(
            [self.d_hidden, four_hidden],
            Some(self.d_hidden),
            Some(four_hidden),
            device,
        );
        // Bias layout along the last dim: [i | f | g | o], each `d_hidden` wide.
        // Setting the `f` slice to 1.0 is the Jozefowicz init trick.
        let bias = if self.bias {
            let zeros: Param<Tensor<1>> = Initializer::Zeros.init_with(
                [four_hidden],
                Some(self.d_input),
                Some(four_hidden),
                device,
            );
            if self.forget_gate_bias_one {
                // `detach()` strips the autodiff graph node so `Param::from_tensor`
                // (which calls `require_grad()` and panics on non-leaf tensors)
                // can treat the result as a fresh leaf parameter. No-op on
                // devices without autodiff enabled.
                let tensor = zeros
                    .val()
                    .slice_fill(self.d_hidden..2 * self.d_hidden, 1.0f32)
                    .detach();
                Some(Param::from_tensor(tensor))
            } else {
                Some(zeros)
            }
        } else {
            None
        };

        FusedLstm {
            w_ih,
            w_hh,
            bias,
            d_input: self.d_input,
            d_hidden: self.d_hidden,
        }
    }
}

/// LSTM with concatenated 4-gate weights.
///
/// Gate ordering along the last dim of `w_ih` / `w_hh` / `bias` is
/// `[i, f, g, o]` (PyTorch / cuDNN convention), so the weights are
/// transferable to/from any such implementation.
#[derive(Module, Debug)]
pub struct FusedLstm {
    /// Input projection weights, shape `[d_input, 4 * d_hidden]`.
    pub w_ih: Param<Tensor<2>>,
    /// Hidden projection weights, shape `[d_hidden, 4 * d_hidden]`.
    pub w_hh: Param<Tensor<2>>,
    /// Optional bias, shape `[4 * d_hidden]`.
    pub bias: Option<Param<Tensor<1>>>,
    /// Input feature size.
    pub d_input: usize,
    /// Hidden state size.
    pub d_hidden: usize,
}

impl FusedLstm {
    /// Forward pass over a full sequence.
    ///
    /// - `input`: `[batch, seq_len, d_input]`
    /// - `state`: optional initial `(hidden, cell)` each shaped `[batch, d_hidden]`
    ///
    /// Returns `(output, final_state)` where `output` is `[batch, seq_len, d_hidden]`.
    pub fn forward(
        &self,
        input: Tensor<3>,
        state: Option<FusedLstmState<2>>,
    ) -> (Tensor<3>, FusedLstmState<2>) {
        let (h0, c0) = match state {
            Some(s) => (Some(s.hidden), Some(s.cell)),
            None => (None, None),
        };
        let bias = self.bias.as_ref().map(Param::val);

        // The specialised path covers the shape `SequenceModel` always uses:
        // a bias and no caller-supplied initial state. It is what keeps the
        // autodiff graph O(1) in sequence length instead of O(seq_len), so the
        // fallback below is correctness insurance for unusual call sites, not
        // a path training is expected to take.
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(bias) = bias.clone()
            && h0.is_none()
            && c0.is_none()
        {
            let (output, final_state) = super::backend::dispatch_fused_lstm_forward(
                input,
                self.w_ih.val(),
                self.w_hh.val(),
                bias,
            );
            let FusedLstmStateOut { hidden, cell } = final_state;
            return (output, FusedLstmState::new(cell, hidden));
        }

        let (output, final_state) =
            fused_lstm_forward(input, self.w_ih.val(), self.w_hh.val(), bias, h0, c0);

        let FusedLstmStateOut { hidden, cell } = final_state;
        (output, FusedLstmState::new(cell, hidden))
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;

    /// Gradient parity: the single-op BPTT backward pass must produce the same
    /// gradients as per-op autodiff tracking.
    ///
    /// This is the test that matters for [`super::backend::FusedLstmOp`]. That
    /// backward pass is hand-written, so nothing but a numerical comparison
    /// against the reference catches an error in it — and a wrong gradient does
    /// not crash, it just trains the model slightly wrongly for weeks.
    ///
    /// `FusedLstm::forward` routes through the backend extension (the single
    /// tracked op); [`fused_lstm_forward`] called directly stays on plain
    /// tensor ops, so autodiff tracks every step of it. The two must agree.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn single_op_backward_matches_per_op_reference() {
        let device = Device::flex().autodiff();
        let batch = 2;
        let seq = 6;
        let d_input = 3;
        let d_hidden = 4;

        let input_values: Vec<f32> = (0..batch * seq * d_input)
            .map(|i| ((i as f32 * 0.23).cos() - 0.1) * 0.5)
            .collect();
        let make_input = || {
            Tensor::<3>::from_data(
                TensorData::new(input_values.clone(), [batch, seq, d_input]),
                &device,
            )
            .require_grad()
        };

        let fused = FusedLstmConfig::new(d_input, d_hidden, true).init(&device);

        // Fast path: through the extension, so one tracked op.
        let fast_input = make_input();
        let (fast_output, _) = fused.forward(fast_input.clone(), None);
        let fast_grads = fast_output.powf_scalar(2.0).sum().backward();

        // Reference: the same weights driven through the plain-tensor forward,
        // with every step tracked individually.
        let reference_input = make_input();
        let w_ih = Tensor::<2>::from_data(fused.w_ih.val().into_data(), &device).require_grad();
        let w_hh = Tensor::<2>::from_data(fused.w_hh.val().into_data(), &device).require_grad();
        let bias = Tensor::<1>::from_data(
            fused.bias.as_ref().expect("bias enabled").val().into_data(),
            &device,
        )
        .require_grad();

        let (reference_output, _) = fused_lstm_forward(
            reference_input.clone(),
            w_ih.clone(),
            w_hh.clone(),
            Some(bias.clone()),
            None,
            None,
        );
        let reference_grads = reference_output.powf_scalar(2.0).sum().backward();

        let max_abs_difference = |left: Vec<f32>, right: Vec<f32>, name: &str| {
            assert_eq!(left.len(), right.len(), "{name}: gradient length mismatch");
            let difference = left
                .iter()
                .zip(right.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                difference < 1e-4,
                "{name}: single-op backward disagrees with the per-op reference \
                 (max abs diff = {difference:.3e})"
            );
        };

        fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
            tensor.into_data().try_to_vec::<f32>().unwrap()
        }

        max_abs_difference(
            values(fast_input.grad(&fast_grads).expect("fast input gradient")),
            values(
                reference_input
                    .grad(&reference_grads)
                    .expect("reference input gradient"),
            ),
            "d_input",
        );
        max_abs_difference(
            values(
                fused
                    .w_ih
                    .val()
                    .grad(&fast_grads)
                    .expect("fast w_ih gradient"),
            ),
            values(
                w_ih.grad(&reference_grads)
                    .expect("reference w_ih gradient"),
            ),
            "d_w_ih",
        );
        max_abs_difference(
            values(
                fused
                    .w_hh
                    .val()
                    .grad(&fast_grads)
                    .expect("fast w_hh gradient"),
            ),
            values(
                w_hh.grad(&reference_grads)
                    .expect("reference w_hh gradient"),
            ),
            "d_w_hh",
        );
        max_abs_difference(
            values(
                fused
                    .bias
                    .as_ref()
                    .expect("bias enabled")
                    .val()
                    .grad(&fast_grads)
                    .expect("fast bias gradient"),
            ),
            values(
                bias.grad(&reference_grads)
                    .expect("reference bias gradient"),
            ),
            "d_bias",
        );
    }

    /// Basic shape and finiteness check.
    #[test]
    fn forward_shape_and_state() {
        let device = Device::flex();
        let fused = FusedLstmConfig::new(3, 4, true).init(&device);
        let input: Tensor<3> = Tensor::zeros([2, 5, 3], &device);
        let (out, state) = fused.forward(input, None);
        assert_eq!(out.dims(), [2, 5, 4]);
        assert_eq!(state.hidden.dims(), [2, 4]);
        assert_eq!(state.cell.dims(), [2, 4]);
        assert!(
            out.to_data()
                .try_to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite()),
            "output contains non-finite values"
        );
    }

    /// BPTT reaches every parameter and the input.
    ///
    /// The cell math is tracked per op by autodiff rather than by a
    /// hand-written backward pass, so this guards the thing that can still
    /// break: a parameter dropping out of the graph (an un-tracked `val()`,
    /// a stray `detach()`) and silently training at zero gradient.
    #[test]
    fn backward_reaches_every_parameter() {
        let device = Device::flex().autodiff();
        let batch = 2;
        let seq = 6;
        let d_input = 3;
        let d_hidden = 4;

        let input_data: Vec<f32> = (0..batch * seq * d_input)
            .map(|i| ((i as f32 * 0.23).cos() - 0.1) * 0.5)
            .collect();
        let input =
            Tensor::<3>::from_data(TensorData::new(input_data, [batch, seq, d_input]), &device)
                .require_grad();

        let fused = FusedLstmConfig::new(d_input, d_hidden, true).init(&device);

        let (output, _) = fused.forward(input.clone(), None);
        let gradients = output.powf_scalar(2.0).sum().backward();

        let finite_and_nonzero = |values: Vec<f32>, name: &str| {
            assert!(
                values.iter().all(|v| v.is_finite()),
                "{name} gradient has non-finite values"
            );
            assert!(
                values.iter().any(|v| *v != 0.0),
                "{name} gradient is all zeros, so it is detached from the graph"
            );
        };

        // The gradients differ in rank, so each is reduced to its values here
        // rather than collected into one array.
        finite_and_nonzero(
            input
                .grad(&gradients)
                .expect("input gradient")
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            "input",
        );
        finite_and_nonzero(
            fused
                .w_ih
                .val()
                .grad(&gradients)
                .expect("w_ih gradient")
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            "w_ih",
        );
        finite_and_nonzero(
            fused
                .w_hh
                .val()
                .grad(&gradients)
                .expect("w_hh gradient")
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            "w_hh",
        );
        finite_and_nonzero(
            fused
                .bias
                .as_ref()
                .expect("bias enabled")
                .val()
                .grad(&gradients)
                .expect("bias gradient")
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            "bias",
        );
    }
}
