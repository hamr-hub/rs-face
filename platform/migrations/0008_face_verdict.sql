-- Per-face liveness verdict label needed to calibrate: threshold tuning must
-- separate faces that were judged real from those blocked as spoof. Stays
-- NULL when liveness did not run for that face (mirrors the quality columns).
ALTER TABLE faces ADD COLUMN IF NOT EXISTS is_real BOOLEAN;
ALTER TABLE faces ADD COLUMN IF NOT EXISTS blocked BOOLEAN NOT NULL DEFAULT false;
