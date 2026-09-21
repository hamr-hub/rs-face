-- rs-face Platform schema (v5) — persons + person_faces gallery tables.
--
-- Naming: the existing `faces` table is per-job detection results
-- (each row = one face detected in one frame of a job). The new
-- "registered gallery faces" domain is distinct, so we use a separate
-- table name `person_faces` rather than reusing `faces` — the index
-- columns differ, the cascade graph differs, and conflating them
-- would corrupt the existing per-job face telemetry.
--
-- A `person` is a real-world identity (one human). A `person_face` is
-- one enrolled photo / crop of that person, with the embedding that
-- the runtime uses for 1:N identification. The face row also carries
-- an optional storage pointer to the original image so the platform
-- can rebuild embeddings later (e.g. after a model upgrade) without
-- forcing the operator to re-upload.
--
-- Embeddings are stored as `BYTEA` (f32 little-endian packed). The dim
-- is fixed at 128 by the bundled EmbedNet model and indexed via a
-- hash on the centroid for fast gallery lookups. A future migration
-- can extend the dim without breaking on-disk rows: the `dim` column
-- documents what each row was stored as.

CREATE TABLE IF NOT EXISTS persons (
    id           TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    external_id  TEXT,
    note         TEXT,
    -- Per-person aggregated metadata. The runtime does not read this;
    -- it exists so the UI can show "5 faces enrolled, last seen
    -- 2026-09-20" without an aggregate query over `person_faces`.
    faces_count  INTEGER NOT NULL DEFAULT 0,
    created_ms   BIGINT NOT NULL,
    updated_ms   BIGINT NOT NULL,
    archived     BOOLEAN NOT NULL DEFAULT false,
    archived_ms  BIGINT
);
CREATE UNIQUE INDEX IF NOT EXISTS persons_external_id_uidx
    ON persons(external_id) WHERE external_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS persons_archived_idx
    ON persons(archived, created_ms DESC);

CREATE TABLE IF NOT EXISTS person_faces (
    id           TEXT PRIMARY KEY,
    person_id    TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    -- Centroid hash: sha256(embedding bytes) truncated to 16 hex chars.
    -- Lets the runtime quickly decide "this probe might match person X
    -- — load their embeddings" without a full scan. Not unique —
    -- distinct embeddings can collide on the prefix; the per-face
    -- compare is the source of truth.
    centroid     TEXT NOT NULL,
    -- Embedding dim (always 128 for the default EmbedNet, but recorded
    -- so a model swap doesn't silently miscompare rows).
    dim          INTEGER NOT NULL,
    -- The 128-f32 little-endian embedding blob. Stored as BYTEA so a
    -- future schema-migration can change dim without ALTER TYPE.
    embedding    BYTEA NOT NULL,
    -- Source pointer (S3 key under rsface bucket, or local media key).
    source_key   TEXT,
    bbox_x       INTEGER NOT NULL,
    bbox_y       INTEGER NOT NULL,
    bbox_w       INTEGER NOT NULL,
    bbox_h       INTEGER NOT NULL,
    quality      REAL NOT NULL,
    created_ms   BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS person_faces_person_idx ON person_faces(person_id);
CREATE INDEX IF NOT EXISTS person_faces_centroid_idx ON person_faces(centroid);

-- Bumps the per-person face counter atomically. Implemented as a
-- trigger so application code can't forget to keep it in sync with
-- INSERT/DELETE.
CREATE OR REPLACE FUNCTION persons_touch_faces_count() RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE persons SET faces_count = faces_count + 1, updated_ms = EXTRACT(EPOCH FROM now()) * 1000
            WHERE id = NEW.person_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE persons SET faces_count = faces_count - 1, updated_ms = EXTRACT(EPOCH FROM now()) * 1000
            WHERE id = OLD.person_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS persons_faces_count_sync_ins ON person_faces;
CREATE TRIGGER persons_faces_count_sync_ins
    AFTER INSERT ON person_faces
    FOR EACH ROW EXECUTE FUNCTION persons_touch_faces_count();

DROP TRIGGER IF EXISTS persons_faces_count_sync_del ON person_faces;
CREATE TRIGGER persons_faces_count_sync_del
    AFTER DELETE ON person_faces
    FOR EACH ROW EXECUTE FUNCTION persons_touch_faces_count();
