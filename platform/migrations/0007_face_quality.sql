-- Per-face liveness quality measurements used to calibrate replay thresholds.
-- Populated when the liveness feature ran for that face; stay NULL otherwise
-- (so older rows and non-liveness jobs are distinguishable from a real 0).
ALTER TABLE faces ADD COLUMN IF NOT EXISTS sharpness       REAL;
ALTER TABLE faces ADD COLUMN IF NOT EXISTS mean_brightness REAL;
ALTER TABLE faces ADD COLUMN IF NOT EXISTS clipped_ratio   REAL;
ALTER TABLE faces ADD COLUMN IF NOT EXISTS high_freq_ratio REAL;
