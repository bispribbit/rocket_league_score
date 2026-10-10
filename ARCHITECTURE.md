# Architecture

## Crates

```
crates/
├── replay_structs/          # Shared types: frames, players, ranks, dataset splits
├── replay_parser/           # .replay → frames (positions, boost, inputs, demos, scoreboard) via boxcars
├── feature_extractor/       # frames → 71 per-player stats (whole match or a frame window)
├── skill_model/             # Gradient-boosted trees, timeline, roast, compact bundle format (WASM-safe)
├── skill_model_training/    # extract_stats + train binaries; GBDT trainer and evaluation
├── is_this_a_smurf/         # Dioxus WASM web app (embeds data/skill_model.bin)
├── ballchasing_downloader/  # Ballchasing API ingestion, download loop, DB maintenance binaries
├── database/                # PostgreSQL access via SQLx (replays, players, splits)
└── config/                  # Environment and replay-store configuration
```

## Data flow

```
Ballchasing API ──► ballchasing_downloader ──► PostgreSQL (replays, player ranks, splits)
                                         └──► replay files on disk

replay files + ranks ──► extract_stats ──► data/match_stats.csv, data/window_stats.csv
                                              │
                                              ▼
                                    train ──► data/skill_model.bin
                                              │ (include_bytes!)
                                              ▼
user's .replay ──► replay_parser ──► feature_extractor ──► skill_model ──► is_this_a_smurf UI
```

The app and training share one code path for stats (`feature_extractor`) and prediction
(`skill_model`), so what is evaluated is exactly what ships.

## Key decisions

- **Hand-written stats + gradient-boosted trees** instead of a neural network. Faster to
  train (minutes on CPU), deterministic, small enough for the browser, and far more accurate
  on this data; see [`docs/model.md`](docs/model.md).
- **Team-canonical stats.** Every positional stat is measured from the player's own side of
  the field, using the roster slot's team, so blue and orange players are comparable.
- **Lobby level + gap to the lobby.** Two ensembles per model: one for the lobby level, one
  for each player's deviation from it.
- **Whole match for the verdict, 60-second windows for the timeline.**
- **Labels** are each player's ranked-3v3 rank at match time (from Ballchasing), converted to
  the middle of the division's MMR range.

## Database

- **replays**: id, file path, playlist, rank band, download status, dataset split
- **replay_players**: per-replay player name, team, rank and division

Train/evaluation splits are assigned once and stored, so every evaluation uses the same
held-out replays.
