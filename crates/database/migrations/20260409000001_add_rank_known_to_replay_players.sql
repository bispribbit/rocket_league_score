-- Add rank_known column to distinguish players whose rank was reported by the API
-- from players whose rank was inferred from the lobby min/max midpoint.
ALTER TABLE replay_players ADD COLUMN rank_known BOOLEAN NOT NULL DEFAULT true;

-- Backfill: mark players whose rank in the metadata was null.
-- A player's rank is "known" only if their individual rank object is non-null
-- in the ballchasing metadata JSON.
UPDATE replay_players rp
SET rank_known = false
WHERE NOT EXISTS (
    SELECT 1
    FROM replays r,
    LATERAL (
        SELECT p->>'name' AS player_name, p->'rank' AS rank_info
        FROM jsonb_array_elements(
            COALESCE(r.metadata->'blue'->'players', '[]'::jsonb)
            || COALESCE(r.metadata->'orange'->'players', '[]'::jsonb)
        ) p
    ) all_players
    WHERE r.id = rp.replay_id
      AND all_players.player_name = rp.player_name
      AND all_players.rank_info IS NOT NULL
      AND all_players.rank_info != 'null'::jsonb
);
