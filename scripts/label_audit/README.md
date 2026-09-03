# Label audit — performance-based vs account-rank labels

Answers a question no metric in `docs/experiment.md` up to row 29 could: is the
detector actually bad, or is it being scored against the wrong label?

Every metric through row 29 defines a "smurf proxy" as a player whose **account rank**
sits ≥150 MMR above their lobby's median rank. The product question is different —
*"this silver lobby has one guy playing like a platinum and exterminating the
competition"* — which is a claim about how someone **played**. The two disagree in both
directions, and both disagreements were invisible:

* a high-ranked player who had an ordinary game is a positive with nothing to detect;
* a lower-ranked player who genuinely dominated is a **negative**, though they are
  exactly what the product wants surfaced.

## Producing the inputs

```sh
# per-player performance, model-independent (~90 s for the 3,013 evaluation replays)
cargo run --release --example match_performance -- \
    --split evaluation --out data/eval_match_performance.csv

# per-player predictions for whichever checkpoint you are auditing
cargo run --release --example revalidate -- \
    --model models/<arm>/checkpoint_best_ordinal --split evaluation \
    --dump-predictions data/<arm>_eval_predictions.csv
```

Both are keyed by `(replay_id, slot)` under the same slot convention (blue sorted by
name → 0–2, orange → 3–5), so the join is exact.

## Running the audit

```sh
gawk -f join_predictions_performance.awk \
     data/eval_match_performance.csv data/<arm>_eval_predictions.csv > joined.tsv

gawk -f average_precision_by_label.awk joined.tsv        # AP under 3 label definitions
gawk -f per_lobby_pick_accuracy.awk  data/eval_match_performance.csv data/<arm>_eval_predictions.csv
gawk -f mixed_lobby_chance_baseline.awk data/eval_match_performance.csv data/<arm>_eval_predictions.csv
```

## The three labels

| label | definition |
|---|---|
| `acct` | account rank ≥150 MMR above lobby median — the label used through row 29 |
| `top`  | highest ball-possession share in the lobby — pure performance |
| `both` | account-positive **and** top possession — "high-ranked *and* actually dominating" |

## Caveats

* Rows where the in-replay player name did not match the database name are dropped
  (`matched=0`, ~4 %). Keeping them would score missing data as "played terribly" and
  bias the audit in exactly the direction being tested.
* `both` has only 85 positives on the evaluation split, so its lift carries real sampling
  noise. Treat differences under ~20 % as inconclusive.
* AP needs no fit/held-out split — it is a property of a ranking, not of a fitted
  threshold — so these numbers are computed over the whole joined set. They are therefore
  **not** directly comparable to `fit_threshold`'s held-out-half AP.
