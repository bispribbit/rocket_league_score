-- Sizing queries for trajectory-mined smurf labels.
--
-- Background: the training labels call a smurf by their (low) displayed rank, which is
-- the error we want to detect. Mixed-rank lobbies give us a *proxy* — a correctly
-- labelled strong player — but not a real positive. Rank trajectories can give real
-- positives: an account whose rank climbs implausibly fast is a smurf under the
-- operational definition, and its early matches are genuine positives.
--
-- Whether that is worth building depends entirely on how densely players recur across
-- the ~30k sampled replays. These two queries answer that before any code is written.
--
-- Run:  psql "$DATABASE_URL" -f scripts/smurf-trajectory-sizing.sql
--
-- CAVEAT: replay_players has no platform ID, only player_name, so these counts merge
-- distinct people who share a display name and split anyone who renamed. Treat the
-- numbers as an upper bound on usable trajectories.

\echo '== A. Player recurrence across replays =='

WITH appearances AS (
    SELECT player_name, COUNT(*) AS matches
    FROM replay_players
    GROUP BY player_name
)
SELECT COUNT(*)                                  AS distinct_players,
       COUNT(*) FILTER (WHERE matches >= 2)      AS seen_2_plus,
       COUNT(*) FILTER (WHERE matches >= 5)      AS seen_5_plus,
       COUNT(*) FILTER (WHERE matches >= 10)     AS seen_10_plus,
       ROUND(AVG(matches), 2)                    AS mean_matches,
       MAX(matches)                              AS max_matches
FROM appearances;

\echo '== B. Rank climbs among players with 3+ dated matches =='

-- division_index is the 1-based position in the rank_division enum, so the difference
-- between two indices is a count of divisions climbed. 'unranked' is excluded.
WITH observed AS (
    SELECT rp.player_name,
           array_position(enum_range(NULL::rank_division), rp.rank_division) AS division_index,
           (r.metadata ->> 'date')::timestamptz                              AS played_at
    FROM replay_players rp
    JOIN replays r ON r.id = rp.replay_id
    WHERE rp.rank_known
      AND rp.rank_division <> 'unranked'
      AND r.metadata ->> 'date' IS NOT NULL
),
spans AS (
    SELECT player_name,
           COUNT(*)                                                            AS matches,
           MAX(division_index) - MIN(division_index)                           AS division_gain,
           EXTRACT(EPOCH FROM (MAX(played_at) - MIN(played_at))) / 86400.0     AS span_days
    FROM observed
    GROUP BY player_name
    HAVING COUNT(*) >= 3
)
SELECT COUNT(*)                                                                AS players_3plus_dated,
       COUNT(*) FILTER (WHERE division_gain >= 4)                              AS gained_4_plus_div,
       COUNT(*) FILTER (WHERE division_gain >= 8)                              AS gained_8_plus_div,
       COUNT(*) FILTER (WHERE division_gain >= 8  AND span_days <= 30)         AS gained_8_plus_in_30d,
       COUNT(*) FILTER (WHERE division_gain >= 12 AND span_days <= 30)         AS gained_12_plus_in_30d,
       ROUND(AVG(span_days)::numeric, 1)                                       AS mean_span_days
FROM spans;

\echo '== C. Sample of the fastest climbers (manual sanity check) =='

WITH observed AS (
    SELECT rp.player_name,
           array_position(enum_range(NULL::rank_division), rp.rank_division) AS division_index,
           (r.metadata ->> 'date')::timestamptz                              AS played_at
    FROM replay_players rp
    JOIN replays r ON r.id = rp.replay_id
    WHERE rp.rank_known
      AND rp.rank_division <> 'unranked'
      AND r.metadata ->> 'date' IS NOT NULL
)
SELECT player_name,
       COUNT(*)                                                          AS matches,
       MAX(division_index) - MIN(division_index)                         AS division_gain,
       ROUND((EXTRACT(EPOCH FROM (MAX(played_at) - MIN(played_at))) / 86400.0)::numeric, 1) AS span_days
FROM observed
GROUP BY player_name
HAVING COUNT(*) >= 3
ORDER BY division_gain DESC, matches DESC
LIMIT 25;
