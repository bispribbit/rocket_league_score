# Overfit harness (`overfit_wgpu`) experiments

For a **self-contained handoff** (goals, commands, file map, conventions), see [`overfit_wgpu.md`](overfit_wgpu.md).

All runs use **burn-wgpu** with the FusedLSTM / CubeCL path, default `lr=1e-2`, `seq_len=300`, `batch_size=32` unless stated otherwise. Pass criteria: **T1** final RMSE < 50 MMR, **T2** final RMSE < 300 MMR, **T3** per-rank **SSL** RMSE < 350 MMR.

| # | Date (approx) | Description | T1 (final RMSE) | T2 (final RMSE) | T3 (overall RMSE) | T3 (SSL RMSE) | Pass |
|---|----------------|------------|-----------------|-----------------|-------------------|----------------|------|
| 1 | 2026-04-25 | **Baseline harness:** sequential indices `(0..n)`, vanishing MSE on normalised targets, `forward` only, no rank weights / pinball / oversampling. | 22.9 | 791.3 | 550.1 | 1367.7 | T1 only |
| 2 | 2026-04-25 | **Phase A + B (harness only):** `build_oversampled_indices` per epoch, pinball (same constants as `training.rs`), inverse-frequency `compute_inverse_frequency_weights` + `lookup_rank_weights`, `forward_with_lobby_scale` alternating 1.0 / 0.0 per batch, stretched warmup and cosine `lr` floor 0.05, collapse-diag logging. *No* `LOBBY_BIAS_OUTPUT_SCALE` yet; *i.i.d.* label jitter not in this row (see 3). | 33.7 (early stop) | 1284.9 | 553.6 | 1297.2 | T1 only |
| 3 | 2026-04-25 | **Phase C:** `LOBBY_BIAS_OUTPUT_SCALE = 0.2` on lobby bias in `SequenceModel` (`ml_model`), **mean-zero** label jitter in `training.rs` and the same in `overfit_wgpu` (loss on jittered targets, RMSE on clean targets). | 25.3 (early stop) | 1285.0 | 554.0 | *not in log snippet* | T1 only |
| 4 | 2026-04-25 | **`--mse-only`:** MSE on jittered targets, **no** pinball, **unit** rank weights; same oversampling / lobby alternation / schedule / mean-zero jitter / `LOBBY_BIAS_OUTPUT_SCALE`. | 2.1 (early stop) | 1033.0 | 548.8 | 1377.7 | T1 only |
| 5 | 2026-04-25 | **More steps + higher LR:** `--mse-only --lr 3e-2 --t2-epochs 200 --t3-epochs 250` (put `--epochs` before per-tier overrides if you use both). | 44.2 (early stop) | 320.3 | 549.1 | 1388.4 | T1 only |
| 6 | 2026-04-25 | **T2 pass recipe:** `--mse-only --lr 3e-2 --t2-epochs 270 --t3-epochs 250 --lr-floor 0.10`. T2 early-stops ~139 when RMSE &lt; 300. | 23.6 (early stop) | 96.0 (early stop) | 432.5 | 1128.5 | T1+T2 |
| 7 | 2026-04-26 | **T1+T2+T3 pass** (harness + model): `LOBBY_BIAS_OUTPUT_SCALE = 0`, `LABEL_JITTER_STD = 40` (harness + `training.rs`), `Wgpu::<f32>::seed(&device, 42)` for reproducibility, T3-only MSE row boost (mean target &gt; 1800 MMR → 10× weight), T3 uses `1.5×` base LR and `max(0.15, --lr-floor)` cosine floor. CLI: `--mse-only --lr 3e-2 --t2-epochs 270 --t3-epochs 450 --lr-floor 0.10`. | 27.9 (early stop) | 184.9 (early stop) | 228.4 | 230.4 | **All pass** |
| 8 | 2026-04-26 | **Production parity refactor:** `minibatch_loss.rs` — `train()` and the default harness call `production_training_minibatch_loss` (Huber, pinball, rank weights, ordinal, pairwise). **`--mse-only`** uses `mse_ablation_minibatch_loss`: `forward_with_lobby_scale` (same MMR path as eval), shared `mean_zero_label_jitter_normalized`, plain MSE (no `0.5` factor), T3 row boost unchanged. Same CLI as row 7. | 17.6 (early stop) | 147.3 (early stop) | 327.2 | 210.5 | **All pass** |

## Notes

- **Row 8:** After extracting shared loss, the first `--mse-only` T3 run regressed to a **constant-MMR** collapse (~976 SSL) until the ablation was aligned with eval: `forward_with_lobby_scale`, shared jitter helper, and **plain** `diff²` (not `0.5·diff²`). Full 450-epoch T3 re-validates row 7 (SSL 210.5 MMR this run).
- **T1** always passes: single SSL replay, single constant label — not a strong test of *using features*, only of the stack and optimisation.
- **T2** got *worse* in scalar RMSE vs row 1 (791 → ~1285) after changing the training signal: the old ~791 is consistent with a **global constant predictor** near the class-weighted mean; the new loss pushes the network away from that collapse but does not yet reach < 300 on **clean-target** RMSE.
- **Row 4:** Removing pinball + rank weights **helped** T2 (1285 → **1033**) and brought `pred_mean` in line with the oversampled batch mean (~1090), but RMSE is still ~**1033** — the network is **not** learning replay-specific scalars for Bronze vs SSL, only a **global level** (within-batch mean). T3 U-shape unchanged vs row 3; SSL still **1377.7**.
- **Row 5:** Higher base LR + more epochs moved T2 to **320.3** (only ~20 MMR over the 300 bar); late training (epochs 180–200) drives most of the gain. T3 / SSL barely moved vs row 4.
- **Row 6:** **`--lr-floor 0.10`** + **`--t2-epochs 270`** clears **T2** early (~epoch 139, RMSE **96**). **T3** finally moves: overall **549→432**, SSL **1388→1128** (still **FAIL** vs 350). Late T3 (epochs 200–250) shows a step down (~550→430) similar in spirit to T2’s late drop.
- **T3** SSL remains the bottleneck; row 6 is the first large SSL improvement in this table. Full per-rank tables were omitted for row 3; re-run to capture SSL and Bronze tails. Append SSL RMSE to row 3 when available.

## Proposed next steps (short list)

1. **Harness objective match:** make T2/T3 early-stop and report **Huber** (or the same combined loss as production) for the *logging* metric, or add a second line “clean Huber / clean RMSE” so we do not tune only to MSE if that is at odds with full training.
2. ~~**T2-only ablation:**~~ Done as row 4 (`--mse-only`).
3. **Learning rate / steps:** T2: row **6** (`--lr 3e-2`, `--lr-floor 0.10`, `--t2-epochs 270`) **PASS**. T3: extend **`--t3-epochs`** and/or tune tail LR — see row **7**.
4. **Lobby / jitter knobs:** try `LOBBY_BIAS_OUTPUT_SCALE` at **0.1** or **0.0** in a dedicated experiment row; try **reducing** `LABEL_JITTER_STD` in the harness if training noise drowns out the tiny batch signal.
5. ~~**Production parity:**~~ Done: shared `minibatch_loss` (`production_training_minibatch_loss` in `train` and default harness); `--mse-only` uses `mse_ablation_minibatch_loss` (see row 8).

Add a new table row (and bump `#`) for every new change or hyperparameter sweep so we do not re-discuss the same dead ends.
