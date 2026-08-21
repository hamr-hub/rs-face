-- rs-face Platform schema (v1)
-- 任务/帧/人脸三张表,涵盖 JobRegistry 当前内存中所有需要持久化的状态。

CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('image', 'video', 'stream')),
    display_name  TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('queued','running','done','cancelled','error')),
    created_ms    BIGINT NOT NULL,
    finished_ms   BIGINT,
    frames_processed  BIGINT NOT NULL DEFAULT 0,
    frames_with_face  BIGINT NOT NULL DEFAULT 0,
    total_detections  BIGINT NOT NULL DEFAULT 0,
    original_key  TEXT,
    error         TEXT,
    -- v2:检测算法名(haar/cnn/yunet/mtcnn/hog)。前端算法过滤 chip 用。
    -- 用 IF NOT EXISTS 兼容老库(虽然 PG 9.6+ 才支持,生产库 PG 17 OK)。
    algo          TEXT
);
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS algo TEXT;
CREATE INDEX IF NOT EXISTS jobs_created_idx ON jobs(created_ms DESC);

CREATE TABLE IF NOT EXISTS frames (
    job_id        TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    idx           BIGINT NOT NULL,
    timestamp_ms  BIGINT NOT NULL,
    annotated_key TEXT,
    original_key  TEXT,
    PRIMARY KEY (job_id, idx)
);
CREATE INDEX IF NOT EXISTS frames_job_idx ON frames(job_id, idx);

CREATE TABLE IF NOT EXISTS faces (
    job_id        TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    frame_idx     BIGINT NOT NULL,
    face_idx      INTEGER NOT NULL,
    key           TEXT NOT NULL,
    x             INTEGER NOT NULL,
    y             INTEGER NOT NULL,
    w             INTEGER NOT NULL,
    h             INTEGER NOT NULL,
    score         REAL NOT NULL,
    PRIMARY KEY (job_id, frame_idx, face_idx),
    FOREIGN KEY (job_id, frame_idx) REFERENCES frames(job_id, idx) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS faces_job_idx ON faces(job_id);
