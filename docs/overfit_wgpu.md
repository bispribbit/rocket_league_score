# Handoff: `overfit_wgpu` regression harness

Use this document when continuing work without the user in the loop (for example on Windows). The running experiment table lives in [`experiment.md`](experiment.md); append new rows there after each meaningful run.

## What this is

We are stress-testing **Rocket League MMR prediction** training on **burn-wgpu** (Vulkan / DX12 via **wgpu**) with the **FusedLSTM / CubeCL** path. The binary **`overfit_wgpu`** (`crates/ml_model_training/src/bin/overfit_wgpu.rs`) runs three tiers on a **small fixed segment store** so regressions show up quickly.

## Pass criteria (all must pass for exit code 0)

| Tier | Scenario | Metric | Target |
|------|-----------|--------|--------|
| **T1** | Single SSL replay, one constant label | Final **clean-target** RMSE (MMR) | **< 50** |
| **T2** | Bronze-1 vs SSL pair (~90 segments) | Final clean RMSE | **< 300** (early stop after 2 consecutive epochs below) |
| **T3** | Balanced mini-set across ranks (~201 segments) | **SupersonicLegend (SSL)** per-rank RMSE | **< 350** |

Training minimizes **MSE on mean-zero jittered targets** (and optional pinball + rank weights unless `--mse-only`). **Reported RMSE is always on clean (non-jittered) targets** in MMR space.

## Important implementation details

- **Oversampling:** Per-epoch shuffled indices (`build_oversampled_indices`) so batches are rank-balanced and not a fixed `(0..n)` order.
- **Lobby path:** `forward_with_lobby_scale` with alternating lobby scale **1.0 / 0.0** per batch so the model cannot rely only on lobby bias.
- **Phase C (`ml_model`):** `LOBBY_BIAS_OUTPUT_SCALE` scales the lobby bias head (currently **0.2**); reduces “always predict lobby mean” shortcuts.
- **Jitter:** `training.rs` provides **mean-zero** label jitter in normalized space; harness matches it. Loss uses jittered targets; metrics use clean targets.
- **`--mse-only`:** MSE only, **unit** rank weights, **no** pinball. This recipe is what made **T2** reachable (see row 6 in `experiment.md`).
- **Learning rate:** Warmup + cosine decay; effective LR uses **`--lr-floor`** as a **fraction of base `--lr`** (minimum multiplier on the cosine tail). Example: `--lr 3e-2 --lr-floor 0.10` → tail LR **3e-3**.

## Current status (as of row 8 in `experiment.md`)

- **T1, T2, T3:** **All pass** with the row **7** CLI (`--mse-only --lr 3e-2 --t2-epochs 270 --t3-epochs 450 --lr-floor 0.10`) and row **8** implementation: production loss is `production_training_minibatch_loss` in `minibatch_loss.rs` (shared with `train`); the `--mse-only` ablation uses `forward_with_lobby_scale`, shared mean-zero jitter, and plain MSE (see `experiment.md` row 8).

The process exits **non-zero** if any tier fails, even if T1+T2 pass.

## Exact command that passes T1+T2 (row 6)

Default segments directory: `ballchasing/segments/v6` (override with a positional path first).

**Cargo (from repo root):**

```bash
cargo run --release -p ml_model_training --bin overfit_wgpu -- \
  --mse-only --lr 3e-2 --t2-epochs 270 --t3-epochs 250 --lr-floor 0.10
```

**Release binary (Windows example):**

```text
target\release\overfit_wgpu.exe --mse-only --lr 3e-2 --t2-epochs 270 --t3-epochs 250 --lr-floor 0.10
```

**Validated (row 7):** `--t3-epochs 450` with the code changes in row 7 of `experiment.md` clears T3 (SSL RMSE &lt; 350). The harness also sets a fixed `Wgpu` PRNG seed and T3-only loss shaping; see the binary source.

Other flags: `--t1-epochs`, `--epochs` (sets all three tiers; put **before** per-tier overrides if you mix), `--seq-len`, `--lr`, `--lr-floor`, `--mse-only`.

## Key files

| Area | Path |
|------|------|
| Harness binary | `crates/ml_model_training/src/bin/overfit_wgpu.rs` |
| One-minibatch loss (production + `--mse-only` MSE ablation) | `crates/ml_model_training/src/minibatch_loss.rs` |
| Full training loop | `crates/ml_model_training/src/training.rs` |
| Lobby bias scale | `crates/ml_model/src/lib.rs` (`LOBBY_BIAS_OUTPUT_SCALE`) |
| Experiment log | `experiment.md` |

## Repo conventions (`AGENTS.md`)

- English for code and assistant replies.
- Prefer **`cargo clippy --all-features --all-targets`** on the workspace; if the full workspace fails (for example missing `data/v12.mpk` in another crate), use **`cargo clippy -p ml_model_training --all-features --all-targets`** (and `ml_model` if you touch it).
- After edits: **`cargo fmt --all`**.
- Crates use `dependency.workspace = true` in `Cargo.toml`; versions only in the workspace root.

## If you get stuck

1. **T3 SSL still high after more epochs:** try code-level knobs (document in a new `experiment.md` row): lower `LOBBY_BIAS_OUTPUT_SCALE`, reduce label jitter in harness / `training.rs`, or align loss with full `train()` (Huber, etc.).
2. **Burn tensor gotchas:** per-sample weights must match expected rank/shape (for example `[batch, 1]` where the training code expects 2D).
3. **Always** fill in a new numbered row in `experiment.md` after a sweep so the next session does not repeat dead ends.
