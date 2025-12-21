-- Migration: Add tables for replay tracking and ML models

-- Game mode enum for Rocket League
CREATE TYPE game_mode AS ENUM (
    'soccar_3v3',
    'soccar_2v2',
    'soccar_1v1',
    'hoops',
    'rumble',
    'dropshot',
    'snowday'
);

-- Replay metadata (minimal storage as per requirements)
CREATE TABLE replays (
    id UUID PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    game_mode game_mode NOT NULL,
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for querying by game mode
CREATE INDEX idx_replays_game_mode ON replays(game_mode);

-- Player ratings per replay (actual MMR values for training labels)
CREATE TABLE replay_players (
    id UUID PRIMARY KEY,
    replay_id UUID NOT NULL REFERENCES replays(id) ON DELETE CASCADE,
    player_name TEXT NOT NULL,
    team SMALLINT NOT NULL CHECK (team IN (0, 1)),
    skill_rating INTEGER NOT NULL, -- actual MMR (e.g., 1547)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for querying players by replay
CREATE INDEX idx_replay_players_replay_id ON replay_players(replay_id);

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

