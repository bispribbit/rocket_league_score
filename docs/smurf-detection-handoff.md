# Handoff: per-player skill and smurf detection

Written 2026-08-23, continuing from the epoch-255 run of `lstm_v20`; **updated 2026-08-25**
with the results of steps 0–2. Use this to pick the work back up on another machine. The
running experiment table is [`experiment.md`](experiment.md) — rows **20**–**24** are the
changes described here; append a new row for anything further.

> **Read this first (2026-08-25).** Steps 0–2 are done. Three results change how the rest
> of this document should be read — one of them falsifies an assumption it was built on.
>
> 1. **The model does not order lobbymates at chance.** On held-out mixed lobbies it scores
>    `concordance = 0.619` and `top-1 = 34.7 %` against a 16.7 % baseline. The lobby
>    shortcut cannot produce that — it predicts all six slots identically and scores exactly
>    chance — so some per-player signal is already being read.
> 2. **What fails is the flag threshold, not the ordering.** The mean margin between the
>    outlier and the lobby median is **+10 to +18 MMR** against a **+200 MMR** rule, so
>    detection is ~0–3 %. That is **F6**, not F3.
> 3. **Trajectory-mined labels are a no-go** — the fast-climber set is mostly display-name
>    collisions (step 0 below).
>
> Net effect on the plan: step 6's "make the threshold a fitted percentile" is promoted out
> of the conditional tail and is now the cheapest available win. Step 3 remains the right
> go/no-go, but it is no longer testing "is there *any* per-player signal" — it is testing
> whether that signal survives removing lobby context.
>
> **Update 2026-08-26 — step 2.5 is done and it came back negative.** The threshold was the
> cheapest available win and it was not enough. Re-fitting on held-out data gives at best
> 5.7 % precision at 26 % recall (lift 1.63× over a 2.4 % base rate); the shipped `+200`
> rule turns out to be right at *exactly* the base rate. The reason revises F6: the margin
> distribution has plenty of range, but its heavy right tail belongs to the **negatives**,
> so the strictest thresholds score zero true positives. That is a distribution-*shape*
> problem, which no threshold and no monotone recalibration can fix.
>
> **Everything now rests on step 3.** There is no remaining cheap win: the ordering signal
> is real but weak (`conc = 0.619`, lift 1.63×), and the question of whether it is genuinely
> per-player or an artefact of lobby context is the one that decides whether steps 4–8 are
> worth their multi-day training runs at all.

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
| F4 | ~~**Train/inference feature skew.**~~ **FIXED 2026-08-25 (row 24).** Both paths now share `extract_player_centric_frames_with_context`; clock features are always real, and every in-tree inference caller passes its `ParsedReplay`, so goal-replay exclusion and score diff match training too. Bit-identical output is enforced by a mutation-checked regression test. | `feature_extractor/src/lib.rs` |
| F5 | `player_head_out` is `Linear(64→1)` consumed directly as raw MMR. At init it emits O(1); reaching 900 needs weights ~1000× larger, and the cheapest route is inflating the bias to the population mean — a mean-collapse attractor built into the parameterisation. | `ml_model/src/lib.rs` |
| F6 | ~~**No dynamic range left to flag with.**~~ **REVISED 2026-08-26 (row 25): the range exists, it is the *ordering within* it that fails.** Margins reach `+996 MMR`, but the top of the distribution belongs to the negatives (their p99 is `+456` against the positives' `+178`), so the strictest thresholds score *zero* true positives and the shipped rule's precision equals the base rate. Not a range problem and not fixable by recalibration — a distribution-shape problem. Original text: measured margin is `+18 MMR` against the `+200 MMR` rule. MMR spacing is uneven (C1→C2 is 130 MMR; GC3→SSL is 418), so squared error over-weights the top. At epoch 255, bronze-1 has 3,471 labels and receives **1** prediction; SSL has 2,732 and receives 517. Catching a GC in a gold lobby requires emitting a value ~800 above the lobby median, which the current output range cannot express. | `replay_structs/src/rank.rs` |
| F7 | `mixed_rank_eval` cannot prove what it claims — stacked segments carry their original lobby's context in their own features, so the shortcut still answers it. Becomes valid once the skill tower is self-only. | `bin/mixed_rank_eval.rs` |
| F8 | Cross-lobby pairs are abundant supervision and need no identity: any two players from any two lobbies form a ranking pair valid at their own match times. Blocked today only because both vectors carry lobby context. | — |
| F9 | 1,994 s/epoch × 256 epochs ≈ **6 days per configuration**. Half the rows in `experiment.md` are "pending re-train". | — |

Full write-up with reasoning:
<https://claude.ai/code/artifact/417e3017-0fab-4431-9e93-c9d02011e4d8>

## Corrections to earlier conclusions

- **Row 15's top-1 rate (13–18 %) was not measured the way it was reported.** `max_by`
  returns the *last* maximum, so under a collapsed model `argmax_slot` was always slot 5
  and the metric reduced to "is the outlier the alphabetically-last orange player". Fixed
  in row 21. **Re-run in row 24: the corrected figure is 22.2 % against 16.7 % chance.**
- **Row 15 also ran through the skewed inference path** (F4), so it compared features the
  model was never trained on. **F4 is fixed and row 15 has been re-established (row 24).**
- **The "mean margin ≈ 0 MMR" reading survived, and it turned out to be the whole story.**
  Row 24 measures `+10 MMR` and row 23 `+18 MMR`, against a `+200 MMR` flag rule.
- **"The model orders lobbymates at chance" did *not* survive.** Row 23 measures
  `concordance = 0.619` and `top-1 = 34.7 %` on held-out mixed lobbies. Several claims in
  the diagnosis below were written before this was measurable; where they assume ordering
  is at chance, prefer the measurement.
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

### 0. Run the sizing query — ✅ DONE 2026-08-25, verdict **no-go**

```bash
psql "$DATABASE_URL" -f scripts/smurf-trajectory-sizing.sql
```

**Recurrence is adequate; identity is not.** 108,011 distinct names, 20,304 seen ≥2×, 4,303
≥5×, 1,608 ≥10×. Among 3,849 players with 3+ dated matches, 92 gained ≥8 divisions in 30
days and 39 gained ≥12.

**The 39 candidates are mostly name collisions.** `rank_division` has 86 values, so section
C's top entries (gains of 55–83) span nearly the entire ladder, and the names are `L`,
`000000`, `?`, `-`, empty, runs of `*`, and bare first names. Quantified two ways:

- **18 of the 39 (46 %)** show a ≥4-division rank span **within a single calendar day**;
  11 (28 %) show ≥8. `Wyzerer` gains 15 divisions in 0.1 days.
- Across all 4,168 name-weeks with ≥3 matches, **100 span ≥12 divisions inside one week** —
  a larger impossible-set than the entire candidate set.

**Decision: drop trajectory mining, lean on mixed-rank proxies.** This is the branch the
document pre-committed to. The blocker is schema, not method — `replay_players` has no
platform ID, so identity cannot be resolved at all. **Storing the Ballchasing platform ID
on ingest is the prerequisite for ever revisiting this**, and remains worth doing.

### 1. Re-score the epoch-255 checkpoint — ✅ DONE 2026-08-25

New `examples/revalidate.rs` runs a validation pass with frozen weights, calling
`ml_model_training::compute_validation_loss` directly so the numbers cannot drift from an
in-training pass. Segments load **cache-only**; skipped replays are counted and printed.

```bash
cargo run --release --example revalidate -- \
    --model models/lstm_v20/checkpoint_best --split evaluation
```

Full evaluation split, 3,013 replays, **45,805 segments, 0 skipped**. `constant_baseline`
and `lobby_mean_baseline` reproduce the epoch-255 log exactly, confirming the same set.

```
pred_std=432.4 MMR  pearson_r=0.8197
Whole-match (n=2129 lobbies): player=197.9 MMR  lobby=174.1  |  between=161.5  within=115.3
Within-lobby ordering: concordance=0.553 over 5777 pairs (0.500 = chance)
  mixed lobbies (gap>=150): n=175 conc=0.619 scored=157 top1=34.7% detect=3.2% margin=+18 MMR
```

Three things to take from this:

- **Ordering is above chance, clearly so on the lobbies that matter.** `conc=0.619` and
  `top1=34.7 %` vs 16.7 % chance. The shortcut cannot generate this, because a shortcut
  prediction is identical across all six slots and scores exactly chance. **This falsifies
  the assumption written into step 3's framing below.**
- **Detection fails on range, not on order.** `margin=+18 MMR` against a `+200 MMR` rule.
- **Per-segment RMSE has been overstating the product's error by ~135 MMR.** The checkpoint
  was selected on 332.6 MMR per-segment; the whole-match number that deployment actually
  incurs is 197.9 MMR. This strengthens the open decision about checkpoint selection.

### 2. Fix the train/inference skew (F4) — ✅ DONE 2026-08-25

Both paths now run **one** function,
`extract_player_centric_frames_with_context(frames, roster, segment_length, Option<SequenceContext>)`.
`SequenceContext` borrows `goals` / `goal_frames` / `kickoff_frames` off a `ParsedReplay`.

The clock features were the actual defect and are now **always** derived from
`frame.seconds_remaining`. The old inference path zeroed them, emitting
`seconds_remaining_normalized = 0` alongside `is_overtime = 0` — a pair training cannot
produce, since a zero clock always means overtime. Goal-replay exclusion and the
reconstructed score differential are what `context` actually gates.

**The skew turned out to be fully removable rather than merely gateable:** every in-tree
inference caller already held a `ParsedReplay` and was just passing `&parsed.frames`. New
`ExtractedSegmentFeatures::from_parsed` and `predict_player_centric_per_segment_from_parsed`
carry it through; the app, `flag_smurfs`, and `smurf_spotcheck` all moved onto them.

Four regression tests, **mutation-checked** by reintroducing the bug (two fail as intended):
training vs inference-with-context are bit-identical across every frame/slot/feature; goal
frames are provably excluded so the equality test cannot pass vacuously; the impossible
`(0, 0)` clock pair never appears; score-diff is 0 without metadata and non-zero with it.

**Row 15 re-established** on the fixed path:

```bash
cargo run --release --example smurf_spotcheck -- \
    --model models/lstm_v20/checkpoint_best --split evaluation --min-top-gap 150
```

```
[OVERALL] lobbies=223  detection_rate=0.0%  top1_rate=22.2% (chance 16.7%)  mean_margin=10 MMR
```

Above chance, same direction as step 1, and again margin-limited rather than order-limited.

**Loose end worth closing before either number is quoted as *the* top-1 rate:** step 1's
in-training metric says 34.7 % over 157 scored lobbies; this says 22.2 % over 223. The
populations are built differently (`segment_top_gap_mmr` over cached segments vs
`list_mixed_rank_replays`'s coarse-rank SQL) and so is the coverage (cached segments vs
whole re-parsed replays). Reconcile the two lobby sets before trusting either figure.

### 2.5. Re-fit the flag threshold — ✅ DONE 2026-08-26, result **negative**

Done as specified, and the honest answer is that **no threshold on this checkpoint is
shippable**. Row 25 of `experiment.md` has the full write-up.

```bash
cargo run --release --example revalidate -- \
    --model models/lstm_v20/checkpoint_best --split evaluation \
    --dump-predictions data/lstm_v20_eval_predictions.csv
cargo run --release --example fit_threshold -- \
    --predictions data/lstm_v20_eval_predictions.csv
```

Scoring costs ~1 h; fitting off the dump costs milliseconds. `mixed_lobby_cross_check`
re-derives step 1's numbers from the dump and reproduces them exactly (`mixed=175`,
`scored=157`, `top1=34.7 %`, `margin=+18`), so the dump is faithful.

**Two definitional changes** make these numbers mean more than `mixed_detection_rate`:

- **Every player is scored**, not just the top-labelled player of a mixed lobby. That metric
  only ever inspects the one player who is a positive by construction, so it cannot observe
  a false positive and reports no precision at all.
- **Fitted on one set of lobbies, reported on another**, split by replay — margins are
  lobby-relative, so a lobby split across both sides would compute its medians from a subset
  of its own players.

**Results.** The shipped rule over all 9,309 scored players: **flags 185 (1.99 %), 5 correct
— precision 2.7 % against a 2.43 % base rate.** It is not merely too strict; when it fires
it is right at chance. Held-out: **average precision 0.0408 vs 0.0251 base rate, lift
1.63×**. Best operating point ≈ `+32 MMR` → precision 5.7 %, recall 26.1 %, while flagging
11.4 % of everyone.

**This revises F6.** F6 read the `+18 MMR` margin as "no dynamic range left to flag with".
The range exists — margins reach `+996` — but it is **anti-informative at the top**:

| class | p05 | p25 | median | p75 | p95 | p99 |
|---|---|---|---|---|---|---|
| smurf proxy (n=119) | −90 | −9 | **+10** | +36 | +81 | **+178** |
| everyone else (n=4,625) | −59 | −12 | +0 | +12 | +61 | **+456** |

The heaviest right tail belongs to the **negatives**, so the strictest thresholds are the
*least* precise: the top 1 % (`+499`) and top 2 % (`+208`) of players both score **zero**
true positives. The model's largest within-lobby deviations are its largest errors, not its
best detections; the usable signal sits in modest margins (`+20`…`+80`).

**So the binding constraint is the *shape* of the margin distribution, not the threshold
value** — and no monotone recalibration can change a shape. That closes the threshold as a
line of attack and moves the decision entirely onto step 3.

### 3. Go/no-go ablation, scored ordinally

On a fixed dev subset, train the existing architecture on the **self-only 27-feature
slice**, scored by **within-lobby concordance / top-1 on held-out mixed lobbies**, not RMSE.

**Re-framed by step 1.** The original question was "is per-player skill legible at all",
and step 1 answered yes — `conc=0.619` on mixed lobbies is not reachable through the lobby
shortcut. The live question is now narrower and sharper: **how much of that 0.619 survives
when lobby context is removed?**

- Self-only holds near 0.619 → the signal was per-player all along; the two-tower plan is
  sound and F7's objection to `mixed_rank_eval` dissolves.
- Self-only collapses toward 0.500 → the current edge is coming from context the two-tower
  split is designed to strip out, and the real work is per-player feature engineering.

Either way this is still the right next experiment, and still the gate on steps 4–8.
**After step 2.5 came back negative it is also the *only* remaining experiment** — there is
no cheaper move left to try first.

#### 🔴 IN FLIGHT as of 2026-08-26 22:26 EDT

Both arms were launched sequentially and should finish ~08:50 EDT. **Check these before
starting any new work** — if they completed, the go/no-go verdict is sitting in the logs.

| arm | model name | feature view | log |
|---|---|---|---|
| control | `lstm_v22_full` | full-106 | `models/20260826_222608.txt` |
| ablation | `lstm_v22_self` | self-only-27 | the next `models/*.txt` after it |

Both: 3,000-replay dev subset (45,759 segments), 60 epochs, batch 144, `lr=3e-2`,
validation on the whole evaluation split.

```bash
# The verdict. Compare the last one of these from each arm.
grep "Within-lobby ordering" models/20260826_222608.txt | tail -3
```

Read `mixed lobbies (gap>=150): ... conc=`. **Control tells you what this data budget can
reach at all** — it is not expected to match the `0.619` of the 255-epoch full-data run, and
comparing the ablation against `0.619` instead of against its own control would be the easy
mistake here. The comparison that answers step 3 is **ablation vs control**, both at 60
epochs on the same 3,000 replays.

If either arm died, `grep -c "completed in"` its log to see how far it got; `RESUME=true`
with the same `MODEL_NAME` picks up from the last checkpoint (saved every 5 epochs), and
skips warm-start by design.

#### How to run it (machinery landed 2026-08-26)

`FeatureView::SelfOnly` zeroes the 79 context features rather than narrowing the input
tensor. Zero input contributes nothing through the LSTM's input weights, so it removes the
same information a narrower tensor would, but leaves architecture, parameter count and
tensor shapes identical — the two arms then differ only in what the model can see. A
narrower tensor would confound the ablation with a capacity change.

Run **both arms**; the step-1 checkpoint is not a valid control, because it saw all 27k
replays for 255 epochs and these see 3k for 60.

```bash
# Control: same subset, same schedule, full 106-feature view.
MODEL_NAME=lstm_v22_full DEV_SUBSET_REPLAYS=3000 EPOCHS=60 RESUME=false \
    cargo run --release --example pipeline

# Ablation: identical except the context features are zeroed.
MODEL_NAME=lstm_v22_self DEV_SUBSET_REPLAYS=3000 EPOCHS=60 RESUME=false \
    SELF_ONLY_FEATURES=true cargo run --release --example pipeline
```

`DEV_SUBSET_REPLAYS` — **not** `MAX_REPLAYS`, which is a smoke-test mode that leaks
evaluation replays into training and disables validation entirely. The dev subset draws only
from the training split, keeps validation on the **whole** evaluation split (so concordance
is measured over the same 175 mixed lobbies as the `0.619` baseline), and selects replays by
a stable hash so both arms see identical data.

Budget, from the lstm_v20 log: 412,625 training segments → 1,940 s/epoch, validation 54.5 s.
At 3,000 of 27,150 replays that is ≈ 4.5 min/epoch, so **≈ 4.5 h per arm, ~9 h for both**.

**Score it on `concordance` / `mixed top1`, never on RMSE.** Removing the lobby shortcut is
*expected* to make RMSE much worse — the shortcut is worth 98.4 % of that objective. An RMSE
regression here is the ablation working, not the model failing. Read the
`Within-lobby ordering:` line of the validation block, and re-fit the threshold on each
resulting checkpoint with `revalidate --self-only --dump-predictions` → `fit_threshold` to
see whether the margin distribution's shape improved.

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
  **Step 1 strengthens the case:** per-segment RMSE (332.6) overstates the deployed error
  (197.9) by ~135 MMR, so the quantity the run was checkpointed on is not the quantity the
  product incurs — let alone the ordinal quantity it actually needs.
- Store the Ballchasing platform ID on `replay_players`? **Step 0 settles this: yes.** It is
  the sole blocker on trajectory mining, and without it 46 % of the fast-climber candidates
  are provable name collisions. Cheap to add at ingest.
- ~~Is `+200 MMR` the right flag threshold?~~ **Closed by step 2.5 (row 25): no threshold on
  this checkpoint is worth shipping.** The shipped rule's precision equals the base rate, and
  the best held-out operating point reaches only 5.7 % precision at 26 % recall. Re-open this
  only after a model change moves the margin distribution; re-fitting is now a one-command
  operation (`revalidate --dump-predictions` → `fit_threshold`) and should be re-run against
  every future checkpoint rather than re-litigated.
- Reconcile the two mixed-lobby populations (`segment_top_gap_mmr` over cached segments vs
  `list_mixed_rank_replays` coarse-rank SQL); they disagree, 34.7 % vs 22.2 %. **Half-closed
  by step 2.5:** `mixed_lobby_cross_check` reproduces the in-training 34.7 % exactly from the
  prediction dump, so the two figures do not differ in *how* the rate is computed. What
  remains is purely the lobby population, which is now the only place left to look.
- **New:** `max_replays` is a smoke-test mode, not an experiment mode — it samples across
  ranks ignoring `dataset_split` (so evaluation replays can leak into training) and it
  disables validation entirely, which would have silently produced a step-3 run with no
  within-lobby metrics at all. Use `dev_subset_replays` for anything comparable; it draws
  only from the training split, keeps validation on the **whole** evaluation split so
  concordance stays comparable to the `0.619` baseline, and selects by a stable hash so both
  ablation arms see identical data. `max_replays` should probably be deleted.

## Build notes

The `database` crate uses sqlx compile-time macros. Without a reachable Postgres, use the
checked-in offline cache:

```bash
SQLX_OFFLINE=true cargo check --workspace --all-targets
SQLX_OFFLINE=true cargo test --workspace
```

On this machine `DATABASE_URL` **is** set and `psql` is available, so the sqlx macros and
the DB-backed examples (`revalidate`, `smurf_spotcheck`) all work directly.

`cargo check --workspace` fails in `is_this_a_smurf` with `Asset at /assets/tailwind.css
doesn't exist` — a missing generated Dioxus asset, unrelated to the ML crates. Scope to the
crates you are working on to avoid it:

```bash
SQLX_OFFLINE=true cargo check -p feature_extractor -p ml_model -p ml_model_training \
    -p rocket_league_score --all-targets
```

**Re-scoring a checkpoint without re-training** (~1 h on CPU for the full evaluation split,
vs ~6 days to re-train):

```bash
cargo run --release --example revalidate -- \
    --model models/lstm_v20/checkpoint_best --split evaluation
```

Remaining clippy warnings in `training.rs` are pre-existing `nursery`/`pedantic` lints
(`suboptimal_flops` on squared-error accumulation, matching the existing pattern at
`accumulate_per_rank_errors`). Zero errors, zero new warning classes.
