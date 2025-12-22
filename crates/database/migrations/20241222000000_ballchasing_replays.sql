-- Migration: Add tables for ballchasing replay downloads

-- Rank enum matching ballchasing API filter options
CREATE TYPE ballchasing_rank AS ENUM (
    'unranked',
    'bronze_1',
    'bronze_2',
    'bronze_3',
    'silver_1',
    'silver_2',
    'silver_3',
    'gold_1',
    'gold_2',
    'gold_3',
    'platinum_1',
    'platinum_2',
    'platinum_3',
    'diamond_1',
    'diamond_2',
    'diamond_3',
    'champion_1',
    'champion_2',
    'champion_3',
    'grand_champion'
);

-- Download status for tracking progress
CREATE TYPE download_status AS ENUM (
    'not_downloaded',
    'in_progress',
    'downloaded',
    'failed'
);

-- Ballchasing replay metadata and download tracking
CREATE TABLE ballchasing_replays (
    id UUID PRIMARY KEY,  -- ballchasing replay ID (from API)
    rank ballchasing_rank NOT NULL,
    metadata JSONB NOT NULL,
    download_status download_status NOT NULL DEFAULT 'not_downloaded',
    file_path TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for querying by rank
CREATE INDEX idx_ballchasing_replays_rank ON ballchasing_replays(rank);

-- Index for querying by download status
CREATE INDEX idx_ballchasing_replays_download_status ON ballchasing_replays(download_status);

-- Composite index for fetching pending downloads by rank
CREATE INDEX idx_ballchasing_replays_rank_status ON ballchasing_replays(rank, download_status);

