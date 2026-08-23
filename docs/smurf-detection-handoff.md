# Handoff: per-player skill and smurf detection

Written 2026-08-23, continuing from the epoch-255 run of `lstm_v20`. Use this to pick the
work back up on another machine. The running experiment table is
[`experiment.md`](experiment.md) — rows **20** and **21** are the changes described here;
append a new row for anything further.

Companion doc: [`overfit_wgpu.md`](overfit_wgpu.md) for the regression harness.

## Goal, stated precisely

**Goal #1 is finding smurfs in lobbies** — players whose real skill sits far above the rank
they are labelled with. Two consequences shape everything below:

1. The product question is **ordinal and within-lobby** ("is this player above their
   lobbymates"), not absolute MMR regression.
2. **A smurf's label is wrong in exactly the direction we want to detect**, so the labels
   are adversarial rather than merely noisy.

A player's skill also drifts over time, so account identity does **not** imply constant
skill. Ranks are recorded per match (from `summary.blue.players[].rank` in the Ballchasing
payload, not a profile lookup), so drift is already represented correctly in the data.

## Diagnosis: why per-player predictions have not worked

From the epoch-255 validation log:

```
pred_std=406.4 MMR  pearson_r=0.7770  constant_baseline=470.8 MMR  lobby_mean_baseline=59.3 MMR
train 302.4 MMR RMSE   valid 332.6 MMR RMSE
```

`lobby_mean_baseline=59.3` against `constant_baseline=470.8` is the whole story:

- **Within-lobby label variance is 1.6 %** of total (`(59.3 / 470.8)² = 0.0159`). The other
  98.4 % is *which lobby this is*.
- Matchmaking builds homogeneous lobbies, so "predict this player's rank" ≈ "predict this
  lobby's rank".
- Every player's 106-feature vector contains the full state of the other five cars, so the
  shortcut — read the lobby, emit it six times — is available and worth 98.4 % of the
  objective.

**This is a specification problem, not an optimisation problem.** No learning rate, tail
boost, or spread penalty fixes a 1.6 % signal share. Rows 11–19 of `experiment.md` are nine
attempts at the optimisation reading.

Secondary findings, in rough priority order:

| # | Finding | Where |
|---|---------|-------|
| F2 | The split encoder in the docs was never built. `SELF_PLAYER_FEATURE_COUNT = 27` is exported but **never read**; both forward passes run one LSTM over all 106 features. `LOBBY_BIAS_OUTPUT_SCALE = 0.0` also makes the lobby head and its 20 % `lobby_scale` dropout inert. There is currently no lobby/player decomposition anywhere. | `ml_model/src/lib.rs`, `feature_extractor/src/lib.rs:40` |
| F3 | The pairwise hinge is the only within-lobby ordering term: weight 0.1, masks pairs closer than 25 MMR (adjacent divisions sit ~19 apart), and saturates at a flat 50 MMR margin. The spread term compares std across the **whole minibatch**, so between-lobby spread satisfies it. | `minibatch_loss.rs` |
| F4 | **Train/inference feature skew.** The inference path passes `FrameGameContext::default()` and skips goal-replay exclusion. Three features are hard zero, in a combination that never occurs in training (`seconds_remaining=0` always implies `is_overtime=1`). | `feature_extractor/src/lib.rs:672` vs `:891` |
| F5 | `player_head_out` is `Linear(64→1)` consumed directly as raw MMR. At init it emits O(1); reaching 900 needs weights ~1000× larger, and the cheapest route is inflating the bias to the population mean — a mean-collapse attractor built into the parameterisation. | `ml_model/src/lib.rs` |
| F6 | **No dynamic range left to flag with.** MMR spacing is uneven (C1→C2 is 130 MMR; GC3→SSL is 418), so squared error over-weights the top. At epoch 255, bronze-1 has 3,471 labels and receives **1** prediction; SSL has 2,732 and receives 517. Catching a GC in a gold lobby requires emitting a value ~800 above the lobby median, which the current output range cannot express. | `replay_structs/src/rank.rs` |
| F7 | `mixed_rank_eval` cannot prove what it claims — stacked segments carry their original lobby's context in their own features, so the shortcut still answers it. Becomes valid once the skill tower is self-only. | `bin/mixed_rank_eval.rs` |
| F8 | Cross-lobby pairs are abundant supervision and need no identity: any two players from any two lobbies form a ranking pair valid at their own match times. Blocked today only because both vectors carry lobby context. | — |
| F9 | 1,994 s/epoch × 256 epochs ≈ **6 days per configuration**. Half the rows in `experiment.md` are "pending re-train". | — |

Full write-up with reasoning:
<https://claude.ai/code/artifact/417e3017-0fab-4431-9e93-c9d02011e4d8>

## Corrections to earlier conclusions

- **Row 15's top-1 rate (13–18 %) was not measured the way it was reported.** `max_by`
  returns the *last* maximum, so under a collapsed model `argmax_slot` was always slot 5
  and the metric reduced to "is the outlier the alphabetically-last orange player". Fixed
  in row 21; **re-run before citing.** The mean margin of ~0 MMR does not depend on the
  argmax and probably survives.
- **Row 15 also ran through the skewed inference path** (F4), so it compared features the
  model was never trained on. Re-establish it after F4 is fixed.
- An earlier draft of F8 proposed a same-player consistency loss. That assumes stable skill
  and is **wrong**; identity is useful for trajectory mining instead (see below).

## What changed in the code (rows 20–21, this session)

All committed together. No training behaviour changed — these are measurement and
integrity fixes.

**Whole-match + within-lobby validation metrics (row 20)**

- `segment_cache.rs` — `SegmentEntry` gains `replay_id` (from `SegmentFileInfo`, carried
  through `subset_by_indices`); new `SegmentStore::get_replay_id`; `segment_top_gap_mmr`
  and `MIXED_TOP_GAP_THRESHOLD_MMR` made `pub`.
- `training.rs` — new `WithinLobbyAccumulator` regroups validation predictions by replay
  and reports **per-player-per-match RMSE** (median over that player's segments),
  **per-lobby RMSE**, a **between/within decomposition**, **within-lobby pairwise
  concordance** (0.500 = chance), and on mixed lobbies (gap ≥ 150) **top-1 / detection /
  mean margin** under the shipped rule. Exposed as `ValidationLossResult::within_lobby` →
  `TrainingState::last_validation_within_lobby`.
- Top-1 gives **fractional credit on ties**, so a collapsed model scores exactly 1/6
  instead of 100 %.
- `lobby_mean_baseline` deliberately left on the **mean** so it stays comparable with rows
  11–19; the new metrics use the median, where robustness to a smurf matters.
- 5 unit tests in `training.rs`: perfect ordering, collapsed-model-scores-chance,
  between/within separation, segment-median robustness, homogeneous/unknown-slot exclusion.

**Rule unification and spot-check fixes (row 21)**

- `ml_model::SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR` is now the single definition of the
  `+200 MMR` flag margin. The app's badge, its verdict copy, `smurf_spotcheck`, and the
  training metric all alias it.
- `smurf_spotcheck` — fractional top-1 credit, a `tie` state per row, `(chance 16.7%)`
  printed in the summary, and slot mapping now from `PlayerRoster::from_frames` (the same
  roster the prediction path builds) rather than from frame 0, which only agreed when all
  six players were present in that frame.

**New:** `scripts/smurf-trajectory-sizing.sql` (see next step 0).

## Next steps

Ordered so each result is interpretable before the next begins. Steps 0–2 need no GPU
training.

### 0. Run the sizing query — blocked on DB access, do this first

Could not run in the last session: `.env` had only `BALLCHASING_API_KEY`, no
`DATABASE_URL`, and no `psql`/docker client was available.

```bash
psql "$DATABASE_URL" -f scripts/smurf-trajectory-sizing.sql
```

Three sections: player recurrence across replays, division-climb distribution among
players with 3+ dated matches, and the 25 fastest climbers for eyeballing.

**Why it matters:** the mixed-rank lobbies are *proxy* positives — the strong player's
label is correct, so it is a party, not a smurf. Real positives can only come from rank
trajectories (an account climbing implausibly fast is a smurf by the operational
definition) or manual review. If players recur densely enough, trajectory mining gives
genuine labels and a defensible detection rate. If the fastest climbers look like name
collisions, drop the idea and lean on proxies.

Caveat baked into the file: `replay_players` stores only `player_name`, no platform ID, so
counts merge same-named players and split renames — treat as an upper bound. Storing the
Ballchasing platform ID on ingest is worth doing regardless.

### 1. Re-score the epoch-255 checkpoint

Run a validation pass with existing weights (not a fresh train) to see the row-20 metrics
for the first time:

```
Whole-match (per-player median over a replay's segments, n=… lobbies):
  player=… MMR  lobby=… MMR  |  between=…  within=…
Within-lobby ordering: concordance=… over … pairs (0.500 = chance)  |
  mixed lobbies (gap>=150): n=… conc=… scored=… top1=…% detect=…% margin=… MMR
```

Read `within` and `concordance` first — those are the smurf-relevant numbers. A
`concordance` near 0.500 means the model orders lobbymates at chance regardless of how good
the RMSE looks.

### 2. Fix the train/inference skew (F4), then re-run the spot-check

Make `extract_player_centric_game_sequence` and `..._inference` one function with optional
metadata, gating the affected features off in training when metadata is absent.

```bash
cargo run --release --example smurf_spotcheck -- \
  --model models/lstm_v20/checkpoint_best --split evaluation --min-top-gap 150
```

This re-establishes row 15 on a feature path the model has actually seen, with the row-21
metric fixes in place.

### 3. Go/no-go ablation, scored ordinally

On a fixed dev subset, train the existing architecture on the **self-only 27-feature
slice**, scored by **within-lobby concordance / top-1 on held-out mixed lobbies**, not RMSE.

- Clears chance → per-player skill is legible from one car's own kinematics; the two-tower
  plan is sound.
- Does not → no architecture saves it, and the real work is per-player feature engineering.

**Nothing below is worth doing before this answer is known.**

### 4–8. Conditional on step 3

4. **Head parameterisation (F5)** — predict in normalised space, scale by `MMR_SCALE` on
   output, initialise the output bias to the training mean. Do this before the architecture
   change so the two are not confounded.
5. **Two-tower model (F2 + F3)** — skill tower on self-only features, lobby tower on
   context, `prediction = lobby_level + (skill − lobby_median(skill))`. The centered
   residual **is** the smurf score, trained directly rather than subtracted out afterwards.
   Add Huber on centered residuals and a gap-proportional pairwise margin.
6. **Restore output range (F6)** — rank-index or percentile targets, or promote the
   existing `ordinal_head` to primary. Note: post-hoc monotone recalibration is worth ~36
   MMR of RMSE but leaves within-lobby ordering **provably unchanged**, so it does nothing
   for goal #1 except move the `+200` threshold — which should become a fitted percentile
   anyway.
7. **Cross-lobby ranking supervision (F8)** — valid only once the skill tower is self-only.
8. **Asymmetric trimmed loss** — drop the top 1–2 % of *signed positive* residuals per
   batch (predicted far above label = smurf candidate). Signed, not absolute: a player
   predicted far below their label is a bad game and that gradient should be kept. This is
   the resurrected `SMURF_MASK_START_EPOCH` idea done per-slot; gate it on healthy
   `pred_std`, keep the trim fraction near the assumed base rate, and watch mean prediction
   bias. The trimmed set is the first real smurf candidate list.

**Running in parallel regardless:** keep `fetch_mixed_rank` going. Row 16 settled that
mid-ladder gap ≥ 150 lobbies are ~2 % of ranked-standard and the constraint is the 200/hr
download quota, not availability. 2,412 is thin for the only data that directly supervises
within-lobby separation.

### Longer bets

- **Aggregate during training, not just inference** — pool segment embeddings across a
  whole match and put the loss on the match-level prediction. Matches deployment and
  averages out label noise.
- **Match-level per-player features** — the five cumulative features reset every segment, so
  they measure 20 seconds of boost usage. Boost efficiency and time at zero, supersonic
  fraction, ground/air/wall split, touch quality, rotation consistency, overcommit rate,
  powerslide and air-roll usage, and the variance of each. Per-player by construction and
  not readable off the lobby.
- **LayerNorm** — there is none anywhere in the stack.
- **Fix experiment velocity (F9)** — a fixed dev subset (~20k segments, stratified, mixed
  lobbies over-represented) with a fixed eval protocol. Separately try `seq_len=150` at
  `FRAME_SUBSAMPLE_RATE=4`; if the metric barely moves that is a 4× speedup.

## Open decisions

- Should checkpoint selection and the collapse cutoff move onto `mixed_top1_rate` /
  `concordance` instead of per-segment loss? The plumbing is in place
  (`TrainingState::last_validation_within_lobby`); nothing gates on it yet, deliberately.
- Store the Ballchasing platform ID on `replay_players`? Needed for trustworthy trajectory
  mining; cheap to add at ingest.
- Is `+200 MMR` the right flag threshold, or should it become a percentile fitted on
  held-out mixed lobbies? The latter is more defensible and is what the calibration note in
  `ml_model` points at.

## Build notes

The `database` crate uses sqlx compile-time macros. Without a reachable Postgres, use the
checked-in offline cache:

```bash
SQLX_OFFLINE=true cargo check --workspace --all-targets
SQLX_OFFLINE=true cargo test --workspace
```

Remaining clippy warnings in `training.rs` are pre-existing `nursery`/`pedantic` lints
(`suboptimal_flops` on squared-error accumulation, matching the existing pattern at
`accumulate_per_rank_errors`). Zero errors, zero new warning classes.
