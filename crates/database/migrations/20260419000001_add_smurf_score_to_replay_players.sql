-- Smurf-detection score per player slot.
--
-- After running flag_smurfs.rs, this column holds:
--   predicted_mmr  - mmr_middle(rank_division)
--
-- A large positive value (e.g. > 500) suggests the player performed
-- significantly above their stated rank, flagging a potential smurf.
-- NULL means the player has not yet been scored.
ALTER TABLE replay_players
    ADD COLUMN IF NOT EXISTS smurf_score REAL;
