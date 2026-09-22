//! Person / face gallery: persistence + 1:1 + 1:N identification.
//!
//! The runtime uses the in-tree zero-dep EmbedNet
//! (`rsface::embednet::EmbedNet::new(seed)`). That network is randomly
//! initialised, so its cosine-similarity scores are NOT comparable to a
//! pretrained ArcFace — but they ARE consistent: a probe's cosine to a
//! face enrolled at the same seed is meaningful as a "rank within this
//! session" signal. Same shape as the LBPH / Eigenface gallery used by
//! `crate::recognition`: zero external model files, consistent within a
//! deployment, tunable via `RSFACE_GALLERY_SEED` env.
//!
//! The DB schema lives in `migrations/0005_persons.sql`.

use crate::jobs::DetectorKind;
use crate::liveness::{Liveness, LivenessVerdict};
use crate::persist::Db;
use rsface::embedding::{Embedding, Gallery, MatchConfig};
use rsface::embednet::EmbedNet;
use rsface::face::Detection;
use rsface::image::codec::read_pgm;
use rsface::image::png::decode_to_gray;
use rsface::image::GrayImage;
use rsface::image::RgbImage;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

const EMBED_DIM: usize = 128;

/// One persisted gallery state, kept in memory and synced with the DB.
#[derive(Clone)]
pub struct GalleryState {
    gallery: Arc<RwLock<Gallery>>,
    embedder: Arc<EmbedNet>,
    db: Db,
    cfg: MatchConfig,
    liveness: Option<Liveness>,
    liveness_enforce: bool,
}

impl GalleryState {
    pub async fn load(db: Db, cfg: &crate::config::Config) -> Self {
        let seed = cfg.gallery_seed;
        let match_cfg = cfg.gallery_match_config();
        let embedder = Arc::new(EmbedNet::new(seed));
        let mut gallery = Gallery::new(match_cfg.clone());
        if let Err(e) = hydrate_from_db(&db, &mut gallery).await {
            tracing::warn!("[gallery] hydrate failed: {e} (starting empty)");
        }
        let liveness = Liveness::open(cfg);
        Self {
            gallery: Arc::new(RwLock::new(gallery)),
            embedder,
            db,
            cfg: match_cfg,
            liveness,
            liveness_enforce: cfg.liveness_enforce,
        }
    }

    pub fn cfg(&self) -> &MatchConfig {
        &self.cfg
    }

    /// Whether a liveness backend was successfully loaded.
    #[allow(dead_code)] // used by integration tests and external consumers
    pub fn has_liveness(&self) -> bool {
        self.liveness.is_some()
    }

    /// Run the liveness check for one detection; `None` when no backend is
    /// loaded or no verdict could be produced.
    pub fn check_liveness(&self, rgb: &RgbImage, det: &Detection) -> Option<LivenessVerdict> {
        self.liveness.as_ref().and_then(|live| live.check(rgb, det))
    }

    /// Whether spoof detections should be treated as blocked (enforcement).
    pub fn liveness_enforce(&self) -> bool {
        self.liveness_enforce
    }

    /// Minimal empty instance, mainly for integration tests that build a
    /// JobRegistry by hand (no DB / no async): in-memory gallery, seeded
    /// EmbedNet, no pool. Not used by production startup.
    #[allow(dead_code)]
    pub fn empty_for_tests(seed: u64, cfg: MatchConfig) -> Self {
        Self {
            gallery: Arc::new(RwLock::new(Gallery::new(cfg.clone()))),
            embedder: Arc::new(EmbedNet::new(seed)),
            db: Db { pool: None },
            cfg,
            liveness: None,
            liveness_enforce: false,
        }
    }

    pub async fn identity_count(&self) -> usize {
        self.gallery.read().await.len()
    }

    pub async fn face_count(&self) -> usize {
        self.gallery.read().await.embedding_count()
    }

    /// Embed a single GrayImage crop into a 128-d vector.
    pub fn embed(&self, crop: &GrayImage) -> Option<Embedding> {
        self.embedder.embed(crop)
    }

    /// Detect faces + embed + 1:N rank against the gallery.
    ///
    /// `detector` is **cloned** per call. The DetectorKind wrapper
    /// carries a `CnnScratchInner: !Sync` (the CNN scratch is single-
    /// threaded); if we passed `&DetectorKind` the resulting future
    /// would not be `Send` (axum 0.8 handler bound) and the whole
    /// router would refuse to compile. Cloning is cheap — DetectorKind
    /// is `Arc`-backed internally.
    pub async fn recognize(
        &self,
        detector: DetectorKind,
        rgb: &RgbImage,
        gray: &GrayImage,
        top_k: usize,
    ) -> Vec<RankedFace> {
        let detections = detector.detect(gray);
        let gallery = self.gallery.read().await;
        detections
            .iter()
            .map(|det| {
                let crop = crop_gray(gray, det);
                let liveness = self.liveness.as_ref().and_then(|live| live.check(rgb, det));
                let blocked =
                    self.liveness_enforce && liveness.as_ref().is_some_and(|v| !v.is_real);
                let ranked = if blocked {
                    Vec::new()
                } else {
                    self.embedder
                        .embed(&crop)
                        .map(|emb| gallery.rank(&emb).into_iter().take(top_k).collect())
                        .unwrap_or_default()
                };
                RankedFace {
                    detection: det.clone(),
                    matches: ranked,
                    liveness,
                    blocked,
                }
            })
            .collect()
    }

    /// 1:1 compare: returns the best cosine similarity between the probe
    /// and any of the named identity's enrolled embeddings.
    pub async fn verify(&self, gray: &GrayImage, label: &str) -> Option<f32> {
        let emb = self.embedder.embed(gray)?;
        let gallery = self.gallery.read().await;
        gallery
            .identities()
            .iter()
            .find(|id| id.label == label)
            .map(|id| {
                id.embeddings
                    .iter()
                    .filter_map(|e| emb.cosine(e))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
    }
}

/// One row in a recognition response.
#[derive(Clone, Debug)]
pub struct RankedFace {
    pub detection: Detection,
    /// Top-k ranked identities with cosine similarity, descending.
    pub matches: Vec<(String, f32)>,
    /// Optional silent liveness verdict; `None` when liveness is disabled
    /// or the backend could not produce a decision.
    pub liveness: Option<LivenessVerdict>,
    /// True when liveness enforcement refused to return matches for this face.
    pub blocked: bool,
}

impl Serialize for RankedFace {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("RankedFace", 8)?;
        st.serialize_field("x", &self.detection.x)?;
        st.serialize_field("y", &self.detection.y)?;
        st.serialize_field("w", &self.detection.w)?;
        st.serialize_field("h", &self.detection.h)?;
        st.serialize_field("score", &self.detection.score)?;
        st.serialize_field("matches", &self.matches)?;
        st.serialize_field("liveness", &self.liveness)?;
        st.serialize_field("blocked", &self.blocked)?;
        st.end()
    }
}

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
pub struct Person {
    pub id: String,
    pub display_name: String,
    pub external_id: Option<String>,
    pub note: Option<String>,
    pub faces_count: i32,
    pub created_ms: i64,
    pub updated_ms: i64,
    pub archived: bool,
    pub archived_ms: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FaceMeta {
    pub id: String,
    pub person_id: String,
    pub dim: i32,
    pub quality: f32,
    pub bbox_x: i32,
    pub bbox_y: i32,
    pub bbox_w: i32,
    pub bbox_h: i32,
    pub source_key: Option<String>,
    pub created_ms: i64,
    pub centroid: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PersonCreate {
    pub display_name: String,
    #[serde(default)]
    pub external_id: Option<String>,
    #[serde(default)]
    pub note: Option<String>,
}

#[allow(dead_code)] // reserved for the person update endpoint (next iteration)
#[derive(Clone, Debug, Deserialize)]
pub struct PersonUpdate {
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub external_id: Option<String>,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FaceEnroll {
    /// Optional client-supplied face box (x, y, w, h). When `None` the
    /// handler picks the largest detection in the image.
    #[serde(default)]
    pub bbox: Option<[i32; 4]>,
    /// Optional pointer to the original image in S3 / local. Used so
    /// the gallery can rebuild embeddings later if a model swap
    /// invalidates the on-disk centroid hash.
    #[serde(default)]
    pub source_key: Option<String>,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

impl GalleryState {
    pub async fn list_persons(&self, include_archived: bool) -> Vec<Person> {
        let Some(pool) = self.db.pool.as_ref() else {
            return Vec::new();
        };
        let q = if include_archived {
            "SELECT id, display_name, external_id, note, faces_count, created_ms, updated_ms, archived, archived_ms FROM persons ORDER BY created_ms DESC"
        } else {
            "SELECT id, display_name, external_id, note, faces_count, created_ms, updated_ms, archived, archived_ms FROM persons WHERE archived = false ORDER BY created_ms DESC"
        };
        sqlx::query_as::<
            _,
            (
                String,
                String,
                Option<String>,
                Option<String>,
                i32,
                i64,
                i64,
                bool,
                Option<i64>,
            ),
        >(q)
        .fetch_all(pool)
        .await
        .map(|rows| rows.into_iter().map(row_to_person).collect())
        .unwrap_or_default()
    }

    pub async fn get_person(&self, id: &str) -> Option<Person> {
        let pool = self.db.pool.as_ref()?;
        sqlx::query_as::<_, (String, String, Option<String>, Option<String>, i32, i64, i64, bool, Option<i64>)>(
            "SELECT id, display_name, external_id, note, faces_count, created_ms, updated_ms, archived, archived_ms FROM persons WHERE id = $1",
        )
        .bind(id)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .map(row_to_person)
    }

    pub async fn create_person(&self, body: PersonCreate) -> Result<Person, String> {
        let pool = self
            .db
            .pool
            .as_ref()
            .ok_or_else(|| "DB disabled".to_string())?;
        let id = new_id("p_");
        let now = now_ms();
        sqlx::query(
            "INSERT INTO persons (id, display_name, external_id, note, faces_count, created_ms, updated_ms, archived) \
             VALUES ($1, $2, $3, $4, 0, $5, $5, false)",
        )
        .bind(&id)
        .bind(&body.display_name)
        .bind(&body.external_id)
        .bind(&body.note)
        .bind(now)
        .execute(pool)
        .await
        .map_err(|e| format!("insert person: {e}"))?;
        // In-memory slot — placeholder embedding until first face enrolled.
        let mut g = self.gallery.write().await;
        g.enroll(&id, fallback_embedding());
        drop(g);
        self.get_person(&id)
            .await
            .ok_or_else(|| "person vanished after insert".to_string())
    }

    pub async fn archive_person(&self, id: &str) -> Result<(), String> {
        let pool = self
            .db
            .pool
            .as_ref()
            .ok_or_else(|| "DB disabled".to_string())?;
        let now = now_ms();
        sqlx::query(
            "UPDATE persons SET archived = true, archived_ms = $2, updated_ms = $2 WHERE id = $1",
        )
        .bind(id)
        .bind(now)
        .execute(pool)
        .await
        .map_err(|e| format!("archive person: {e}"))?;
        let mut g = self.gallery.write().await;
        g.remove(id);
        Ok(())
    }

    /// Decode + detect + embed + persist + add to gallery.
    ///
    /// `image_bytes` is the raw upload (PNG / PGM / PPM). The handler
    /// picks the largest face by area if `enroll.bbox` is None; if no
    /// faces are found the function returns `Err("no_face")`.
    pub async fn enroll_face(
        &self,
        person_id: &str,
        image_bytes: &[u8],
        enroll: FaceEnroll,
    ) -> Result<FaceMeta, String> {
        let pool = self
            .db
            .pool
            .as_ref()
            .ok_or_else(|| "DB disabled".to_string())?;
        sqlx::query("SELECT 1 FROM persons WHERE id = $1 AND archived = false")
            .bind(person_id)
            .fetch_optional(pool)
            .await
            .map_err(|e| format!("lookup person: {e}"))?
            .ok_or_else(|| "person_not_found".to_string())?;

        let img = decode_to_gray_or_gray(image_bytes).ok_or_else(|| "decode_failed".to_string())?;
        let detector = build_haar_detector().ok_or_else(|| "detector_build_failed".to_string())?;
        let detections = detector.detect(&img);
        if detections.is_empty() {
            return Err("no_face".to_string());
        }
        let det = match enroll.bbox {
            Some([x, y, w, h]) => detections
                .iter()
                .find(|d| {
                    d.x == x as usize && d.y == y as usize && d.w == w as usize && d.h == h as usize
                })
                .cloned()
                .unwrap_or_else(|| pick_largest(&detections)),
            None => pick_largest(&detections),
        };
        let crop = crop_gray(&img, &det);
        let embedding = self
            .embed(&crop)
            .ok_or_else(|| "embed_failed".to_string())?;

        let face_id = new_id("f_");
        let bytes = embedding_to_bytes(&embedding);
        let centroid = centroid_hash(&bytes);
        let now = now_ms();
        sqlx::query(
            "INSERT INTO person_faces (id, person_id, centroid, dim, embedding, source_key, \
             bbox_x, bbox_y, bbox_w, bbox_h, quality, created_ms) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
        )
        .bind(&face_id)
        .bind(person_id)
        .bind(&centroid)
        .bind(EMBED_DIM as i32)
        .bind(&bytes)
        .bind(&enroll.source_key)
        .bind(det.x as i32)
        .bind(det.y as i32)
        .bind(det.w as i32)
        .bind(det.h as i32)
        .bind(det.score)
        .bind(now)
        .execute(pool)
        .await
        .map_err(|e| format!("insert face: {e}"))?;

        // Update in-memory gallery. The Gallery keeps one Embedding per
        // Identity; we replace so the freshest face wins. (Multi-embedding
        // per identity is supported by rsface::embedding::Identity but
        // would require extending Gallery::enroll; deferred to a future
        // migration so the API stays single-rep per person.)
        {
            let mut g = self.gallery.write().await;
            g.remove(person_id);
            g.enroll(person_id, embedding);
        }
        Ok(FaceMeta {
            id: face_id,
            person_id: person_id.to_string(),
            dim: EMBED_DIM as i32,
            quality: det.score,
            bbox_x: det.x as i32,
            bbox_y: det.y as i32,
            bbox_w: det.w as i32,
            bbox_h: det.h as i32,
            source_key: enroll.source_key,
            created_ms: now,
            centroid,
        })
    }

    pub async fn list_faces(&self, person_id: &str) -> Vec<FaceMeta> {
        let Some(pool) = self.db.pool.as_ref() else {
            return Vec::new();
        };
        sqlx::query_as::<_, (String, String, i32, f32, i32, i32, i32, i32, Option<String>, i64, String)>(
            "SELECT id, person_id, dim, quality, bbox_x, bbox_y, bbox_w, bbox_h, source_key, created_ms, centroid \
             FROM person_faces WHERE person_id = $1 ORDER BY created_ms DESC",
        )
        .bind(person_id)
        .fetch_all(pool)
        .await
        .map(|rows| rows.into_iter().map(row_to_face_meta).collect())
        .unwrap_or_default()
    }

    pub async fn delete_face(&self, face_id: &str) -> Result<(), String> {
        let pool = self
            .db
            .pool
            .as_ref()
            .ok_or_else(|| "DB disabled".to_string())?;
        let row: Option<(String,)> =
            sqlx::query_as("SELECT person_id FROM person_faces WHERE id = $1")
                .bind(face_id)
                .fetch_optional(pool)
                .await
                .map_err(|e| format!("lookup face: {e}"))?;
        let (person_id,) = row.ok_or_else(|| "face_not_found".to_string())?;
        sqlx::query("DELETE FROM person_faces WHERE id = $1")
            .bind(face_id)
            .execute(pool)
            .await
            .map_err(|e| format!("delete face: {e}"))?;
        self.reload_identity_from_db(&person_id).await;
        Ok(())
    }

    /// Rebuild the in-memory `Identity` for `person_id` from the latest
    /// face row. Used after a delete-face so the gallery doesn't keep a
    /// stale embedding.
    async fn reload_identity_from_db(&self, person_id: &str) {
        let Some(pool) = self.db.pool.as_ref() else {
            return;
        };
        let row: Option<(Vec<u8>,)> = sqlx::query_as(
            "SELECT embedding FROM person_faces WHERE person_id = $1 \
             ORDER BY created_ms DESC LIMIT 1",
        )
        .bind(person_id)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();
        let Some((bytes,)) = row else { return };
        let Some(embedding) = Embedding::from_bytes_le(&bytes) else {
            return;
        };
        let mut g = self.gallery.write().await;
        g.remove(person_id);
        g.enroll(person_id, embedding);
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

type PersonRow = (
    String,
    String,
    Option<String>,
    Option<String>,
    i32,
    i64,
    i64,
    bool,
    Option<i64>,
);
type FaceMetaRow = (
    String,
    String,
    i32,
    f32,
    i32,
    i32,
    i32,
    i32,
    Option<String>,
    i64,
    String,
);

fn row_to_person(r: PersonRow) -> Person {
    Person {
        id: r.0,
        display_name: r.1,
        external_id: r.2,
        note: r.3,
        faces_count: r.4,
        created_ms: r.5,
        updated_ms: r.6,
        archived: r.7,
        archived_ms: r.8,
    }
}

fn row_to_face_meta(r: FaceMetaRow) -> FaceMeta {
    FaceMeta {
        id: r.0,
        person_id: r.1,
        dim: r.2,
        quality: r.3,
        bbox_x: r.4,
        bbox_y: r.5,
        bbox_w: r.6,
        bbox_h: r.7,
        source_key: r.8,
        created_ms: r.9,
        centroid: r.10,
    }
}

fn new_id(prefix: &str) -> String {
    use std::hash::{BuildHasher, Hasher, RandomState};
    let mut h = RandomState::new().build_hasher();
    h.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    );
    let s = h.finish();
    format!("{prefix}{s:x}")
}

/// Best-effort GrayImage decoder: PNG via the in-tree codec, PGM/PPM
/// via their respective readers. JPG isn't supported by the zero-dep
/// core; the handler returns `decode_failed`.
pub fn decode_to_gray_or_gray(buf: &[u8]) -> Option<GrayImage> {
    // PNG
    let mut cur = Cursor::new(buf);
    if let Ok(g) = decode_to_gray(&mut cur) {
        return Some(g);
    }
    // PGM
    let mut cur = Cursor::new(buf);
    if let Ok(g) = read_pgm(&mut cur) {
        return Some(g);
    }
    // PPM — read_ppm takes &mut dyn Read; the Cursor implements Read.
    let mut cur = Cursor::new(buf);
    rsface::image::codec::read_ppm(&mut cur)
        .ok()
        .map(|img| img.to_gray())
}

/// Crop the grayscale region described by `detection` from `base`.
pub fn crop_gray(base: &GrayImage, detection: &Detection) -> GrayImage {
    let (width, height) = (base.width(), base.height());
    let x1 = detection.x.min(width);
    let y1 = detection.y.min(height);
    let x2 = (detection.x + detection.w).min(width);
    let y2 = (detection.y + detection.h).min(height);
    let cw = (x2 - x1).max(1);
    let ch = (y2 - y1).max(1);
    let mut out = GrayImage::new(cw, ch);
    let src = base.as_slice();
    let dst = out.as_mut_slice();
    for row in 0..ch {
        let s_off = ((y1 + row).min(height - 1)) * width + x1;
        let d_off = row * cw;
        let len = cw.min(width - x1);
        dst[d_off..d_off + len].copy_from_slice(&src[s_off..s_off + len]);
    }
    out
}

/// Adapter so a `Cursor<&[u8]>` can be passed as `&mut dyn Read` to
/// `read_ppm` (which takes `&mut dyn Read`, not the generic `Cursor`).
#[allow(dead_code)]
struct StdReadAdapter<'a>(&'a mut Cursor<&'a [u8]>);

impl<'a> std::io::Read for StdReadAdapter<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

fn pick_largest(dets: &[Detection]) -> Detection {
    dets.iter()
        .max_by_key(|d| d.w.saturating_mul(d.h))
        .cloned()
        .unwrap_or_else(|| dets[0].clone())
}

fn embedding_to_bytes(e: &Embedding) -> Vec<u8> {
    e.as_slice().iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn centroid_hash(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    hex(&digest[..8])
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn fallback_embedding() -> Embedding {
    Embedding::from_raw(&[1e-6_f32; EMBED_DIM]).expect("constant vector")
}

async fn hydrate_from_db(db: &Db, gallery: &mut Gallery) -> Result<(), String> {
    let Some(pool) = db.pool.as_ref() else {
        return Ok(());
    };
    let rows: Vec<(String, Vec<u8>)> = sqlx::query_as(
        "SELECT person_id, embedding FROM person_faces WHERE person_id IN \
         (SELECT id FROM persons WHERE archived = false) \
         ORDER BY created_ms DESC",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| format!("hydrate: {e}"))?;
    // One Identity per person_id, using the freshest face's embedding.
    let mut seen = std::collections::HashSet::new();
    for (pid, bytes) in rows {
        if !seen.insert(pid.clone()) {
            continue;
        }
        if let Some(e) = Embedding::from_bytes_le(&bytes) {
            gallery.enroll(pid, e);
        }
    }
    Ok(())
}

fn build_haar_detector() -> Option<DetectorKind> {
    // The platform already owns the cascade path; reuse it so the embed
    // path and the recognize path use the same detector configuration.
    let cascade_path = std::env::var("RSFACE_CASCADE").ok()?;
    let bytes = std::fs::read(Path::new(&cascade_path)).ok()?;
    let cascade = rsface::haar::Cascade::load(Path::new(&cascade_path)).ok()?;
    let _ = bytes;
    // DetectorKind is private to jobs.rs, so we re-export a thin shim
    // via jobs::build_detector_by_name.
    let cfg = rsface::detector::DetectorConfig::default();
    let detector = rsface::detector::Detector::new(cascade, cfg);
    Some(crate::jobs::DetectorKind::Haar(detector))
}

// ---------------------------------------------------------------------------
// Embedding LE byte decoding — local helper because rsface::Embedding's
// serde derives don't ship a BYTEA codec.
// ---------------------------------------------------------------------------

trait EmbeddingIo {
    fn from_bytes_le(bytes: &[u8]) -> Option<Embedding>;
}

impl EmbeddingIo for Embedding {
    fn from_bytes_le(bytes: &[u8]) -> Option<Embedding> {
        if !bytes.len().is_multiple_of(4) {
            return None;
        }
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if !v.is_finite() {
                return None;
            }
            out.push(v);
        }
        Embedding::from_raw(&out)
    }
}

// keep `RgbImage` import live for downstream callers that grow this
// file; the unused-import lint otherwise drops it.
#[allow(dead_code)]
fn _rgb_marker(_: &RgbImage) {}
