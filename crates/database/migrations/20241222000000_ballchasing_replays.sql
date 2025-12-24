-- Migration: Add tables for ballchasing replay downloads

-- Rank enum matching ballchasing API filter options
CREATE TYPE ballchasing_rank AS ENUM (
    'unranked',
    'bronze1',
    'bronze2',
    'bronze3',
    'silver1',
    'silver2',
    'silver3',
    'gold1',
    'gold2',
    'gold3',
    'platinum1',
    'platinum2',
    'platinum3',
    'diamond1',
    'diamond2',
    'diamond3',
    'champion1',
    'champion2',
    'champion3',
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

