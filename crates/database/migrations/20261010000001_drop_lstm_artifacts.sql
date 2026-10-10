-- The LSTM model was retired in October 2026 (see docs/model.md). Its checkpoint registry and
-- the per-player smurf scores it wrote are no longer used by any code.
DROP TABLE IF EXISTS models;
ALTER TABLE replay_players DROP COLUMN IF EXISTS smurf_score;
