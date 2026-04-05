-- New enum labels must be committed before use; keep this migration add-only.
ALTER TYPE rank ADD VALUE IF NOT EXISTS 'grand_champion1';
ALTER TYPE rank ADD VALUE IF NOT EXISTS 'grand_champion2';
ALTER TYPE rank ADD VALUE IF NOT EXISTS 'grand_champion3';
ALTER TYPE rank ADD VALUE IF NOT EXISTS 'supersonic_legend';
