-- Add dataset split column to track training vs evaluation replays
-- This allows us to persist the train/test split in the database
-- so training can be resumed and new replays can be added without mixing sets

CREATE TYPE dataset_split AS ENUM ('training', 'evaluation');

ALTER TABLE replays 
ADD COLUMN dataset_split dataset_split;

-- Index for efficient querying by split
CREATE INDEX idx_replays_dataset_split ON replays(dataset_split);

-- Composite index for querying by split and download status
CREATE INDEX idx_replays_split_status ON replays(dataset_split, download_status);

