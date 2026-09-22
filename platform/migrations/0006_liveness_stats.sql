-- Per-job anti-spoofing totals. Populated when the liveness feature is
-- enabled; stay 0 for jobs that never ran liveness.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS spoof_detections BIGINT NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS blocked_detections BIGINT NOT NULL DEFAULT 0;
