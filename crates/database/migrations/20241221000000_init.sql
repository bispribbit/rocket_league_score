-- Migration: Add tables for replay tracking and ML models

-- Game mode enum for Rocket League
CREATE TYPE game_mode AS ENUM (
    'unranked_duels',
    'unranked_doubles',
    'unranked_standard',
    'unranked_chaos',
    'private',
    'season',
    'offline',
    'ranked_duels',
    'ranked_doubles',
    'ranked_solo_standard',
    'ranked_standard',
    'snowday',
    'rocketlabs',
    'hoops',
    'rumble',
    'tournament',
    'dropshot',
    'ranked_hoops',
    'ranked_rumble',
    'ranked_dropshot',
    'ranked_snowday',
    'dropshot_rumble',
    'heatseeker'
);


-- Model versions for tracking trained models
CREATE TABLE models (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    checkpoint_path TEXT NOT NULL,
    training_config JSONB, -- Store training hyperparameters
    metrics JSONB, -- Store training/validation metrics
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Ensure unique versions per model name
    UNIQUE (name, version)
);

-- Index for finding latest model version
CREATE INDEX idx_models_name_version ON models(name, version DESC);


-- Migration: Add tables for replay downloads

-- Rank enum for Rocket League competitive ranks
CREATE TYPE rank AS ENUM (
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

-- Rank division enum with all divisions
CREATE TYPE rank_division AS ENUM (
    'unranked',
    'bronze1_division1',
    'bronze1_division2',
    'bronze1_division3',
    'bronze1_division4',
    'bronze2_division1',
    'bronze2_division2',
    'bronze2_division3',
    'bronze2_division4',
    'bronze3_division1',
    'bronze3_division2',
    'bronze3_division3',
    'bronze3_division4',
    'silver1_division1',
    'silver1_division2',
    'silver1_division3',
    'silver1_division4',
    'silver2_division1',
    'silver2_division2',
    'silver2_division3',
    'silver2_division4',
    'silver3_division1',
    'silver3_division2',
    'silver3_division3',
    'silver3_division4',
    'gold1_division1',
    'gold1_division2',
    'gold1_division3',
    'gold1_division4',
    'gold2_division1',
    'gold2_division2',
    'gold2_division3',
    'gold2_division4',
    'gold3_division1',
    'gold3_division2',
    'gold3_division3',
    'gold3_division4',
    'platinum1_division1',
    'platinum1_division2',
    'platinum1_division3',
    'platinum1_division4',
    'platinum2_division1',
    'platinum2_division2',
    'platinum2_division3',
    'platinum2_division4',
    'platinum3_division1',
    'platinum3_division2',
    'platinum3_division3',
    'platinum3_division4',
    'diamond1_division1',
    'diamond1_division2',
    'diamond1_division3',
    'diamond1_division4',
    'diamond2_division1',
    'diamond2_division2',
    'diamond2_division3',
    'diamond2_division4',
    'diamond3_division1',
    'diamond3_division2',
    'diamond3_division3',
    'diamond3_division4',
    'champion1_division1',
    'champion1_division2',
    'champion1_division3',
    'champion1_division4',
    'champion2_division1',
    'champion2_division2',
    'champion2_division3',
    'champion2_division4',
    'champion3_division1',
    'champion3_division2',
    'champion3_division3',
    'champion3_division4',
    'grand_champion1_division1',
    'grand_champion1_division2',
    'grand_champion1_division3',
    'grand_champion1_division4',
    'grand_champion2_division1',
    'grand_champion2_division2',
    'grand_champion2_division3',
    'grand_champion2_division4',
    'grand_champion3_division1',
    'grand_champion3_division2',
    'grand_champion3_division3',
    'grand_champion3_division4',
    'supersonic_legend'
);

-- Download status for tracking progress
CREATE TYPE download_status AS ENUM (
    'not_downloaded',
    'in_progress',
    'downloaded',
    'failed'
);

-- Replay metadata and download tracking
CREATE TABLE replays (
    id UUID PRIMARY KEY,  -- replay ID (from external API)
    rank rank NOT NULL,
    metadata JSONB NOT NULL,
    download_status download_status NOT NULL DEFAULT 'not_downloaded',
    file_path TEXT NOT NULL,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for querying by rank
CREATE INDEX idx_replays_rank ON replays(rank);

-- Index for querying by download status
CREATE INDEX idx_replays_download_status ON replays(download_status);

-- Composite index for fetching pending downloads by rank
CREATE INDEX idx_replays_rank_status ON replays(rank, download_status);


-- Player ratings per replay (rank division for training labels)
CREATE TABLE replay_players (
    id SERIAL PRIMARY KEY,
    replay_id UUID NOT NULL REFERENCES replays(id) ON DELETE CASCADE,
    player_name TEXT NOT NULL,
    team SMALLINT NOT NULL CHECK (team IN (0, 1)),
    rank_division rank_division NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for querying players by replay
CREATE INDEX idx_replay_players_replay_id ON replay_players(replay_id);