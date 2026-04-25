//! [`FusedLstm`] module. Drop-in replacement for [`burn::nn::Lstm`] using
//! a concatenated 4-gate weight layout.

use burn::config::Config;
use burn::module::{Initializer, Module, Param};
use burn::prelude::*;

use super::backend::{FusedLstmBackend, FusedLstmStateOut};

/// Initial (or returned final) state for a [`FusedLstm`] forward pass.
///
/// Mirrors [`burn::nn::LstmState`] layout so the two modules stay
/// interchangeable at call sites.
#[derive(Debug, Clone)]
pub struct FusedLstmState<B: Backend, const D: usize> {
    /// Cell state.
    pub cell: Tensor<B, D>,
    /// Hidden state.
    pub hidden: Tensor<B, D>,
}

impl<B: Backend, const D: usize> FusedLstmState<B, D> {
    /// Build a new state.
    pub const fn new(cell: Tensor<B, D>, hidden: Tensor<B, D>) -> Self {
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
    /// Weight initializer. Matches [`burn::nn::LstmConfig`]'s default.
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub initializer: Initializer,
    /// Initialise the forget-gate bias slice to `1.0` (zeros elsewhere) when
    /// `bias = true`.
    ///
    /// Jozefowicz et al. (2015) and standard practice in PyTorch /
    /// TensorFlow / JAX — a forget-gate bias of 1 keeps the cell state
    /// flowing at initialisation (`σ(0 + 1) ≈ 0.73` vs. `σ(0) = 0.5`) and
    /// dramatically shortens the early-training phase where the LSTM has
    /// to learn to remember anything.
    #[config(default = "true")]
    pub forget_gate_bias_one: bool,
}

impl FusedLstmConfig {
    /// Initialise a new [`FusedLstm`] module on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> FusedLstm<B> {
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
            let zeros: Param<Tensor<B, 1>> = Initializer::Zeros.init_with(
                [four_hidden],
                Some(self.d_input),
                Some(four_hidden),
                device,
            );
            if self.forget_gate_bias_one {
                // `detach()` strips the autodiff graph node so `Param::from_tensor`
                // (which calls `require_grad()` and panics on non-leaf tensors)
                // can treat the result as a fresh leaf parameter. No-op on
                // non-autodiff backends.
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

/// LSTM with concatenated 4-gate weights, ready for a fused kernel.
///
/// Gate ordering along the last dim of `w_ih` / `w_hh` / `bias` is
/// `[i, f, g, o]` (PyTorch / cuDNN convention), so the weights are
/// transferable to/from any such implementation.
#[derive(Module, Debug)]
pub struct FusedLstm<B: Backend> {
    /// Input projection weights, shape `[d_input, 4 * d_hidden]`.
    pub w_ih: Param<Tensor<B, 2>>,
    /// Hidden projection weights, shape `[d_hidden, 4 * d_hidden]`.
    pub w_hh: Param<Tensor<B, 2>>,
    /// Optional bias, shape `[4 * d_hidden]`.
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// Input feature size.
    pub d_input: usize,
    /// Hidden state size.
    pub d_hidden: usize,
}

impl<B: FusedLstmBackend> FusedLstm<B> {
    /// Forward pass over a full sequence.
    ///
    /// - `input`: `[batch, seq_len, d_input]`
    /// - `state`: optional initial `(hidden, cell)` each shaped `[batch, d_hidden]`
    ///
    /// Returns `(output, final_state)` where `output` is `[batch, seq_len, d_hidden]`.
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: Option<FusedLstmState<B, 2>>,
    ) -> (Tensor<B, 3>, FusedLstmState<B, 2>) {
        let (h0, c0) = match state {
            Some(s) => (Some(s.hidden), Some(s.cell)),
            None => (None, None),
        };

        let (output, final_state) = B::fused_lstm_forward(
            input,
            self.w_ih.val(),
            self.w_hh.val(),
            self.bias.as_ref().map(Param::val),
            h0,
            c0,
        );

        let FusedLstmStateOut { hidden, cell } = final_state;
        (output, FusedLstmState::new(cell, hidden))
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::TensorData;

    use super::*;

    type TestBackend = NdArray;

    /// Basic shape and finiteness check.
    #[test]
    fn forward_shape_and_state() {
        let device = NdArrayDevice::Cpu;
        let fused: FusedLstm<TestBackend> = FusedLstmConfig::new(3, 4, true).init(&device);
        let input: Tensor<TestBackend, 3> = Tensor::zeros([2, 5, 3], &device);
        let (out, state) = fused.forward(input, None);
        assert_eq!(out.dims(), [2, 5, 4]);
        assert_eq!(state.hidden.dims(), [2, 4]);
        assert_eq!(state.cell.dims(), [2, 4]);
        assert!(
            out.to_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite()),
            "output contains non-finite values"
        );
    }

    /// Gradient parity: the custom `Autodiff<NdArray>` backward op must
    /// produce the same input/weight/bias gradients as the default per-op
    /// tracked implementation (the blanket `default_fused_lstm_forward`
    /// that's used when the fast path isn't registered). This catches BPTT
    /// math errors before they reach the GPU path — `Autodiff<NdArray>`
    /// uses the same custom op as `Autodiff<CubeBackend>`, so correctness
    /// here is a strong proxy for the WGPU/CUDA training path.
    #[test]
    fn autodiff_grads_match_reference_bptt() {
        use burn::backend::Autodiff;
        use burn::module::AutodiffModule;

        type AD = Autodiff<NdArray>;

        let device = NdArrayDevice::Cpu;
        let batch = 2;
        let seq = 6;
        let d_input = 3;
        let d_hidden = 4;

        NdArray::<f32>::seed(&device, 7);

        let input_data: Vec<f32> = (0..batch * seq * d_input)
            .map(|i| ((i as f32 * 0.23).cos() - 0.1) * 0.5)
            .collect();
        let input_ad = Tensor::<AD, 3>::from_data(
            TensorData::new(input_data.clone(), [batch, seq, d_input]),
            &device,
        )
        .require_grad();

        let fused_ad: FusedLstm<AD> = FusedLstmConfig::new(d_input, d_hidden, true).init(&device);

        // Copy the same weights into a reference FusedLstm on plain NdArray,
        // and drive both with the default (non-custom-op) forward for the
        // reference gradient. The reference gradient is computed by tracking
        // every per-step op in autograd — slow but trusted.
        let fused_ref_inner = fused_ad.valid();

        // Forward both: tracked fast-path vs reference per-step tracked.
        let (out_fast, _) = fused_ad.forward(input_ad.clone(), None);
        let loss_fast = out_fast.powf_scalar(2.0).sum();
        let grads_fast = loss_fast.backward();

        // For the reference: use Autodiff<NdArray> + the default_fused_lstm_forward
        // path directly. We re-create the module with the same weights, but route
        // through default_fused_lstm_forward by temporarily using a helper.
        let w_ih_v = fused_ad.w_ih.val();
        let w_hh_v = fused_ad.w_hh.val();
        let bias_v = fused_ad.bias.as_ref().unwrap().val();

        let w_ih_ref = Tensor::<AD, 2>::from_data(w_ih_v.into_data(), &device).require_grad();
        let w_hh_ref = Tensor::<AD, 2>::from_data(w_hh_v.into_data(), &device).require_grad();
        let bias_ref = Tensor::<AD, 1>::from_data(bias_v.into_data(), &device).require_grad();
        let input_ref =
            Tensor::<AD, 3>::from_data(TensorData::new(input_data, [batch, seq, d_input]), &device)
                .require_grad();

        let (out_ref, _): (Tensor<AD, 3>, _) =
            crate::fused_lstm::backend::default_fused_lstm_forward::<AD>(
                input_ref.clone(),
                w_ih_ref.clone(),
                w_hh_ref.clone(),
                Some(bias_ref.clone()),
                None,
                None,
            );
        let loss_ref = out_ref.powf_scalar(2.0).sum();
        let grads_ref = loss_ref.backward();

        // Compare input gradient.
        let gi_fast = input_ad.grad(&grads_fast).expect("grad_fast[input]");
        let gi_ref = input_ref.grad(&grads_ref).expect("grad_ref[input]");
        let gi_fast_v: Vec<f32> = gi_fast.into_data().to_vec().unwrap();
        let gi_ref_v: Vec<f32> = gi_ref.into_data().to_vec().unwrap();
        let max_d_input = gi_fast_v
            .iter()
            .zip(gi_ref_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_d_input < 1e-4,
            "d_input mismatch: max abs diff = {max_d_input:.3e}"
        );

        // Compare w_ih / w_hh / bias gradients.
        let gw_ih_fast = fused_ad
            .w_ih
            .val()
            .grad(&grads_fast)
            .expect("grad_fast[w_ih]");
        let gw_ih_ref = w_ih_ref.grad(&grads_ref).expect("grad_ref[w_ih]");
        let max_d_w_ih = gw_ih_fast
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .zip(gw_ih_ref.into_data().to_vec::<f32>().unwrap().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_d_w_ih < 1e-4,
            "d_w_ih mismatch: max abs diff = {max_d_w_ih:.3e}"
        );

        let gw_hh_fast = fused_ad
            .w_hh
            .val()
            .grad(&grads_fast)
            .expect("grad_fast[w_hh]");
        let gw_hh_ref = w_hh_ref.grad(&grads_ref).expect("grad_ref[w_hh]");
        let max_d_w_hh = gw_hh_fast
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .zip(gw_hh_ref.into_data().to_vec::<f32>().unwrap().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_d_w_hh < 1e-4,
            "d_w_hh mismatch: max abs diff = {max_d_w_hh:.3e}"
        );

        let gb_fast = fused_ad
            .bias
            .as_ref()
            .unwrap()
            .val()
            .grad(&grads_fast)
            .expect("grad_fast[bias]");
        let gb_ref = bias_ref.grad(&grads_ref).expect("grad_ref[bias]");
        let max_d_bias = gb_fast
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .zip(gb_ref.into_data().to_vec::<f32>().unwrap().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_d_bias < 1e-4,
            "d_bias mismatch: max abs diff = {max_d_bias:.3e}"
        );

        let _: FusedLstm<TestBackend> = fused_ref_inner;
    }
}
