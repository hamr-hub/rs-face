//! Person / face gallery: persistence + 1:1 + 1:N identification.
//!
//! Pipeline (canonical):
//!
//! ```text
//! RgbImage
//!   └─▶ SCRFD-10G (det_10g.onnx)       : face box + 5 landmarks
//!         └─▶ crate::align::norm_crop    : 112×112 ArcFace canonical
//!               └─▶ ArcFace w600k_r50    : 512-d embedding (InsightFace)
//!                     └─▶ L2-normalise   : cosine-ready
//!                           └─▶ [rsface::embedding::Gallery] rank / identify
//! ```
//!
//! ## Legacy fallback (EmbedNet, 128-d)
//!
//! For historical reasons the very first prototype of this gallery used
//! the in-tree zero-dep EmbedNet (random init, 128-d). That pipeline
//! cannot ship real accuracy — the embeddings live in a different space
//! every session and every restart — so the ArcFace pipeline replaces it
//! everywhere it loads. EmbedNet is still wired in *only* as a soft
//! fallback (`RSFACE_ARCFACE_SOFT_FALLBACK=true`) for environments that
//! cannot ship the ONNX graphs: a server without the graphs stays
//! functional rather than 503ing every /identify call.
//!
//! ## Migration
//!
//! Legacy `person_faces.dim = 128` rows are not consulted by the
//! ArcFace-backed gallery: [`Embedding::cosine`] returns `None` on dim
//! mismatch rather than comparing a prefix, so a 512-d probe simply skips
//! them and the matcher returns `NoCandidates` until the operator
//! re-enrolls those identities. The startup `migrate_drop_legacy_faces` step
//! hard-deletes `dim != 512` rows the first time an ArcFace-backed server
//! boots.
//!
//! The DB schema lives in `migrations/0005_persons.sql`.

#[cfg(feature = "arcface")]
use crate::arcface::ArcFace;
use crate::jobs::DetectorKind;
#[cfg(feature = "arcface")]
use crate::liveness::{Liveness, LivenessVerdict};
use crate::persist::Db;
#[cfg(feature = "arcface")]
use crate::scrfd::Scrfd;
use rsface::embedding::{Embedding, Gallery, MatchConfig};
use rsface::embednet::EmbedNet;
use rsface::face::{Detection, FaceDetection, Landmarks};
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

/// Embedding dimensionality of the standard ArcFace w600k_r50 backbone.
/// Used as the schema-of-record `person_faces.dim` value going forward.
pub const ARCFACE_DIM: usize = 512;

/// `person_faces.dim` value produced by the deprecated zero-dep EmbedNet
/// pipeline (kept so the migration logic can target the right rows).
#[allow(dead_code)]
pub const LEGACY_EMBEDNET_DIM: usize = 128;

/// Which embedder is currently active for a server boot.
#[derive(Clone)]
enum EmbedderKind {
    /// InsightFace ArcFace w600k_r50 + SCRFD-10G. Production pipeline.
    ArcFace {
        arcface: Arc<ArcFace>,
        scrfd: Arc<Scrfd>,
    },
    /// Random-init EmbedNet (128-d) used only when ArcFace graphs are
    /// unavailable AND `RSFACE_ARCFACE_SOFT_FALLBACK=true`.
    Legacy(Arc<EmbedNet>),
}

impl EmbedderKind {
    fn is_arcface(&self) -> bool {
        matches!(self, EmbedderKind::ArcFace { .. })
    }
}

/// One persisted gallery state, kept in memory and synced with the DB.
#[derive(Clone)]
pub struct GalleryState {
    gallery: Arc<RwLock<Gallery>>,
    embedder: EmbedderKind,
    db: Db,
    cfg: MatchConfig,
    liveness: Option<Liveness>,
    liveness_enforce: bool,
}

impl GalleryState {
    pub async fn load(db: Db, cfg: &crate::config::Config) -> Self {
        let match_cfg = cfg.gallery_match_config();
        let mut gallery = Gallery::new(match_cfg.clone());
        // Migration: drop dim != ARCFACE_DIM rows before hydration so the
        // in-memory gallery never sees embeddings that cosine would silently
        // skip. Strict on first boot of an upgraded server; if the operator
        // wants to keep old rows for inspection they must set
        // `RSFACE_ARCFACE_SKIP_MIGRATION=1`.
        if !matches!(
            std::env::var("RSFACE_ARCFACE_SKIP_MIGRATION")
                .ok()
                .as_deref()
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("1") | Some("true") | Some("yes") | Some("on")
        ) {
            if let Err(e) = migrate_drop_legacy_faces(&db).await {
                tracing::warn!("[arcface] migration step failed: {e} (continuing)");
            }
        }
        if let Err(e) = hydrate_from_db(&db, &mut gallery).await {
            tracing::warn!("[gallery] hydrate failed: {e} (starting empty)");
        }
        let liveness = Liveness::open(cfg);
        // Try the industrial pipeline first; on any failure either hard-fail
        // or fall back to EmbedNet (legacy). The soft-fallback keeps dev
        // boxes with no graphs working.
        let embedder = match EmbedderKind::try_open_arcface(cfg).await {
            Ok(kind) => kind,
            Err(e) => {
                if cfg.arcface_soft_fallback {
                    tracing::warn!(
                        "[arcface] pipeline unavailable: {e} — falling back to EmbedNet \
                             (RSFACE_ARCFACE_SOFT_FALLBACK=true). 128-d embeddings will be \
                             written for new faces; this server has no real accuracy."
                    );
                    EmbedderKind::Legacy(Arc::new(EmbedNet::new(cfg.gallery_seed)))
                } else {
                    panic!("[arcface] pipeline unavailable and soft-fallback disabled: {e}");
                }
            }
        };
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

    /// Whether the active embedder is the industrial ArcFace pipeline (vs the
    /// legacy EmbedNet soft-fallback). Used by tests + observability.
    #[allow(dead_code)]
    pub fn is_arcface(&self) -> bool {
        self.embedder.is_arcface()
    }

    /// Minimal empty instance, mainly for integration tests that build a
    /// JobRegistry by hand (no DB / no async): in-memory gallery, seeded
    /// EmbedNet, no pool. Not used by production startup.
    #[allow(dead_code)]
    pub fn empty_for_tests(seed: u64, cfg: MatchConfig) -> Self {
        Self {
            gallery: Arc::new(RwLock::new(Gallery::new(cfg.clone()))),
            embedder: EmbedderKind::Legacy(Arc::new(EmbedNet::new(seed))),
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

    /// Embed a single GrayImage crop into a 128-d vector (legacy path
    /// only). Use the public `/identify` and `/verify` endpoints for the
    /// ArcFace pipeline.
    #[allow(dead_code)]
    pub fn embed(&self, crop: &GrayImage) -> Option<Embedding> {
        match &self.embedder {
            EmbedderKind::Legacy(net) => net.embed(crop),
            EmbedderKind::ArcFace { .. } => None,
        }
    }

    /// Detect faces + embed + 1:N rank against the gallery.
    ///
    /// The caller passes the legacy `DetectorKind` so the platform keeps
    /// one detector construction path. When the active embedder is
    /// ArcFace, this function **additionally** runs SCRFD (whose boxes
    /// carry 5-point landmarks) and picks the SCRFD box whose IoU with
    /// each legacy detection is highest — that gives a stable per-face
    /// landmark feed for the ArcFace alignment while keeping the public
    /// response shape (Detection, not FaceDetection).
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
        // When the ArcFace pipeline is active we use SCRFD's boxes + landmarks
        // directly: SCRFD is more accurate than the platform's Haar cascade
        // and emits the 5-point landmarks the alignment step needs. For the
        // legacy EmbedNet fallback (RSFACE_ARCFACE_SOFT_FALLBACK=true) we
        // keep using the passed-in `detector` so behaviour matches the
        // pre-ArcFace server exactly.
        let legacy_dets = match &self.embedder {
            EmbedderKind::ArcFace { .. } => Vec::new(),
            EmbedderKind::Legacy(_) => detector.detect(gray),
        };
        let scrfd_dets: Option<Vec<FaceDetection>> = match &self.embedder {
            EmbedderKind::ArcFace { scrfd, .. } => match scrfd.detect(rgb) {
                Ok(v) => Some(v),
                Err(e) => {
                    tracing::warn!("[arcface] SCRFD forward failed during recognize: {e:?}");
                    None
                }
            },
            EmbedderKind::Legacy(_) => None,
        };
        let gallery = self.gallery.read().await;

        match &self.embedder {
            EmbedderKind::ArcFace { .. } => {
                let faces = scrfd_dets.unwrap_or_default();
                faces
                    .into_iter()
                    .map(|fd| {
                        let legacy = legacy_box_from_face(&fd);
                        let liveness = self
                            .liveness
                            .as_ref()
                            .and_then(|live| live.check(rgb, &legacy));
                        let blocked =
                            self.liveness_enforce && liveness.as_ref().is_some_and(|v| !v.is_real);
                        let ranked = if blocked {
                            Vec::new()
                        } else if let Some(_landmarks) = fd.landmarks {
                            // Embed this single face via ArcFace.
                            let det_for_rank = Detection {
                                x: legacy.x,
                                y: legacy.y,
                                w: legacy.w,
                                h: legacy.h,
                                score: fd.score,
                            };
                            rank_for_det(&self.embedder, rgb, gray, &det_for_rank, &[fd], |emb| {
                                gallery.rank(emb).into_iter().take(top_k).collect()
                            })
                            .unwrap_or_default()
                        } else {
                            // SCRFD returned a face with no keypoints (rare;
                            // happens on partial detections). Skip — without
                            // landmarks ArcFace alignment is impossible.
                            Vec::new()
                        };
                        RankedFace {
                            detection: legacy,
                            matches: ranked,
                            liveness,
                            blocked,
                        }
                    })
                    .collect()
            }
            EmbedderKind::Legacy(_) => legacy_dets
                .into_iter()
                .map(|det| {
                    let liveness = self
                        .liveness
                        .as_ref()
                        .and_then(|live| live.check(rgb, &det));
                    let blocked =
                        self.liveness_enforce && liveness.as_ref().is_some_and(|v| !v.is_real);
                    let ranked = if blocked {
                        Vec::new()
                    } else {
                        rank_for_det(&self.embedder, rgb, gray, &det, &[], |emb| {
                            gallery.rank(emb).into_iter().take(top_k).collect()
                        })
                        .unwrap_or_default()
                    };
                    RankedFace {
                        detection: det,
                        matches: ranked,
                        liveness,
                        blocked,
                    }
                })
                .collect(),
        }
    }

    /// 1:1 compare: returns the best cosine similarity between the probe
    /// and any of the named identity's enrolled embeddings.
    pub async fn verify(&self, rgb: &RgbImage, gray: &GrayImage, label: &str) -> Option<f32> {
        let emb = match &self.embedder {
            EmbedderKind::ArcFace { arcface, scrfd } => {
                let fd = scrfd.detect(rgb).ok()?;
                let landmarks = pick_landmarks(&fd)?;
                let v = arcface.embed(rgb, &landmarks)?;
                vec_to_embedding(&v)?
            }
            EmbedderKind::Legacy(net) => {
                let fd = detector_for_legacy(gray);
                if fd.is_empty() {
                    return None;
                }
                let det = pick_largest(&fd);
                let crop = crop_gray(gray, &det);
                net.embed(&crop)?
            }
        };
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

impl EmbedderKind {
    /// Try to open SCRFD + ArcFace. `Err` is the "graphs missing or wrong"
    /// signal — the caller decides whether that's fatal.
    #[cfg(feature = "arcface")]
    async fn try_open_arcface(cfg: &crate::config::Config) -> Result<Self, String> {
        let arcface = ArcFace::open(cfg)?
            .ok_or_else(|| "ArcFace weights missing or unreadable".to_string())?;
        let scrfd =
            Scrfd::open(cfg)?.ok_or_else(|| "SCRFD weights missing or unreadable".to_string())?;
        if !arcface.is_standard_dim() {
            return Err(format!(
                "ArcFace graph at {} is not 512-d; only the standard w600k_r50 \
                 backbone is supported for face_db persistence",
                crate::arcface::resolve_arcface_path(cfg).display()
            ));
        }
        tracing::info!(
            "[arcface] pipeline ready: backend={} dim={} standard_dim={}",
            arcface.backend().as_str(),
            512,
            arcface.is_standard_dim()
        );
        Ok(EmbedderKind::ArcFace {
            arcface: Arc::new(arcface),
            scrfd: Arc::new(scrfd),
        })
    }
}

/// Drop any `person_faces` row whose `dim` is not the standard ArcFace
/// dimensionality. This is the migration step that turns a 128-d legacy
/// gallery into one that the ArcFace-backed matcher can read.
///
/// Strict on first boot of an upgraded server. The dry-run setting
/// `RSFACE_ARCFACE_SKIP_MIGRATION=1` skips the deletion (and leaves the
/// legacy rows visible to anyone querying the DB directly).
#[cfg(feature = "arcface")]
async fn migrate_drop_legacy_faces(db: &Db) -> Result<(), String> {
    let pool = db.pool.as_ref().ok_or_else(|| "DB disabled".to_string())?;
    let deleted = sqlx::query("DELETE FROM person_faces WHERE dim != $1")
        .bind(ARCFACE_DIM as i32)
        .execute(pool)
        .await
        .map_err(|e| format!("delete legacy faces: {e}"))?
        .rows_affected();
    if deleted > 0 {
        tracing::info!(
            "[arcface] migrated: deleted {} legacy face row(s) where dim != {}",
            deleted,
            ARCFACE_DIM
        );
    } else {
        tracing::info!("[arcface] migration: no legacy faces found");
    }
    Ok(())
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

        let gray =
            decode_to_gray_or_gray(image_bytes).ok_or_else(|| "decode_failed".to_string())?;
        let rgb = decode_rgb_or_rgb(image_bytes).unwrap_or_else(|| RgbImage::from_gray(&gray));

        // Embedding path branches on the active pipeline:
        //   * ArcFace  → SCRFD gives us a 5-landmark face; align + embed.
        //   * Legacy   → Haar-detect, crop, EmbedNet forward.
        // Either way we also produce a `Detection`-shaped record of the box
        // we used, so the persisted bbox_x/y/w/h + quality stay meaningful.
        let (embedding, face_box) = match &self.embedder {
            EmbedderKind::ArcFace { arcface, scrfd } => {
                let scrfd_dets = scrfd
                    .detect(&rgb)
                    .map_err(|e| format!("SCRFD inference failed during enroll: {e:?}"))?;
                if scrfd_dets.is_empty() {
                    return Err("no_face".to_string());
                }
                let fd = match enroll.bbox {
                    Some([x, y, w, h]) => scrfd_dets
                        .iter()
                        .find(|d| {
                            (d.x1.round() as i32) == x
                                && (d.y1.round() as i32) == y
                                && ((d.x2 - d.x1).round() as i32) == w
                                && ((d.y2 - d.y1).round() as i32) == h
                        })
                        .cloned()
                        .unwrap_or_else(|| pick_largest_face_detection(&scrfd_dets)),
                    None => pick_largest_face_detection(&scrfd_dets),
                };
                let landmarks = fd
                    .landmarks
                    .ok_or_else(|| "SCRFD box had no landmarks; cannot align".to_string())?;
                let raw = arcface
                    .embed(&rgb, &landmarks)
                    .ok_or_else(|| "embed_failed".to_string())?;
                let emb = vec_to_embedding(raw.as_slice())
                    .ok_or_else(|| "ArcFace produced a zero/NaN embedding".to_string())?;
                let det = FaceDetection {
                    x1: fd.x1,
                    y1: fd.y1,
                    x2: fd.x2,
                    y2: fd.y2,
                    score: fd.score,
                    landmarks: fd.landmarks,
                };
                (emb, legacy_box_from_face(&det))
            }
            EmbedderKind::Legacy(net) => {
                let detector =
                    build_haar_detector().ok_or_else(|| "detector_build_failed".to_string())?;
                let detections = detector.detect(&gray);
                if detections.is_empty() {
                    return Err("no_face".to_string());
                }
                let det = match enroll.bbox {
                    Some([x, y, w, h]) => detections
                        .iter()
                        .find(|d| {
                            d.x == x as usize
                                && d.y == y as usize
                                && d.w == w as usize
                                && d.h == h as usize
                        })
                        .cloned()
                        .unwrap_or_else(|| pick_largest(&detections)),
                    None => pick_largest(&detections),
                };
                let crop = crop_gray(&gray, &det);
                let emb = net.embed(&crop).ok_or_else(|| "embed_failed".to_string())?;
                (emb, det)
            }
        };

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
        .bind(embedding.dim() as i32)
        .bind(&bytes)
        .bind(&enroll.source_key)
        .bind(face_box.x as i32)
        .bind(face_box.y as i32)
        .bind(face_box.w as i32)
        .bind(face_box.h as i32)
        .bind(face_box.score)
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
            g.enroll(person_id, embedding.clone());
        }
        Ok(FaceMeta {
            id: face_id,
            person_id: person_id.to_string(),
            dim: embedding.dim() as i32,
            quality: face_box.score,
            bbox_x: face_box.x as i32,
            bbox_y: face_box.y as i32,
            bbox_w: face_box.w as i32,
            bbox_h: face_box.h as i32,
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

/// Best-effort RgbImage decoder. We try PNG first, then PPM (the
/// zero-dep core never learned JPG). The legacy EmbedNet path runs on the
/// GrayImage decoder above and ignores this one.
pub fn decode_rgb_or_rgb(buf: &[u8]) -> Option<RgbImage> {
    // PNG
    let mut cur = Cursor::new(buf);
    if let Ok(img) = rsface::image::png::decode_to_rgb(&mut cur) {
        return Some(img);
    }
    // PPM
    let mut cur = Cursor::new(buf);
    if let Ok(img) = rsface::image::codec::read_ppm(&mut cur) {
        return Some(img);
    }
    None
}

/// Box → landmarks → ArcFace-embed, branch the active embedder.
fn embed_via_active(
    embedder: &EmbedderKind,
    rgb: &RgbImage,
    gray: &GrayImage,
    det: &Detection,
    scrfd_dets: &[FaceDetection],
) -> Option<Embedding> {
    match embedder {
        EmbedderKind::ArcFace { arcface, .. } => {
            let landmarks = pick_landmarks_for(scrfd_dets, det)?;
            arcface
                .embed(rgb, &landmarks)
                .and_then(|v| vec_to_embedding(v.as_slice()))
        }
        EmbedderKind::Legacy(net) => {
            let crop = crop_gray(gray, det);
            net.embed(&crop)
        }
    }
}

/// For each `Detection` (legacy box) pick the SCRFD `FaceDetection` whose
/// box has the highest IoU with it, and return its landmarks. Returns
/// `None` when SCRFD did not produce any face, the legacy box is way off
/// grid, or the matched SCRFD box has no keypoint head.
fn pick_landmarks_for(scrfd_dets: &[FaceDetection], legacy: &Detection) -> Option<Landmarks> {
    let best = scrfd_dets.iter().max_by(|a, b| {
        iou_face_detection(a, legacy)
            .partial_cmp(&iou_face_detection(b, legacy))
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    if iou_face_detection(best, legacy) < 0.05 {
        return None; // legacy box matches nothing — likely a phantom detect
    }
    best.landmarks
}

/// IoU between a `FaceDetection` (x1,y1,x2,y2) and a legacy `Detection`
/// (x,y,w,h). Used to bridge the two coordinate conventions.
fn iou_face_detection(fd: &FaceDetection, legacy: &Detection) -> f32 {
    let a = (fd.x1, fd.y1, fd.x2, fd.y2);
    let b = (
        legacy.x as f32,
        legacy.y as f32,
        (legacy.x + legacy.w) as f32,
        (legacy.y + legacy.h) as f32,
    );
    let ix1 = a.0.max(b.0);
    let iy1 = a.1.max(b.1);
    let ix2 = a.2.min(b.2);
    let iy2 = a.3.min(b.3);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = (a.2 - a.0).max(0.0) * (a.3 - a.1).max(0.0);
    let area_b = (b.2 - b.0).max(0.0) * (b.3 - b.1).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Convert ArcFace's raw `Vec<f32>` output (already L2-normalised by the
/// backbone) into the typed `Embedding`. We re-L2-normalise here as a
/// defence-in-depth measure in case a future model variant ships an
/// unnormalised tensor.
fn vec_to_embedding(v: &[f32]) -> Option<Embedding> {
    Embedding::from_raw(v)
}

/// Build a legacy `Detection` (x,y,w,h,score) from a SCRFD `FaceDetection`
/// for the persisted bbox columns.
fn legacy_box_from_face(fd: &FaceDetection) -> Detection {
    let x = fd.x1.max(0.0).round() as usize;
    let y = fd.y1.max(0.0).round() as usize;
    let w = (fd.x2 - fd.x1).max(0.0).round() as usize;
    let h = (fd.y2 - fd.y1).max(0.0).round() as usize;
    Detection {
        x,
        y,
        w,
        h,
        score: fd.score,
    }
}

/// Largest by area among SCRFD `FaceDetection`s.
fn pick_largest_face_detection(dets: &[FaceDetection]) -> FaceDetection {
    dets.iter()
        .max_by(|a, b| {
            let aa = (a.x2 - a.x1) * (a.y2 - a.y1);
            let bb = (b.x2 - b.x1) * (b.y2 - b.y1);
            aa.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
        .unwrap_or_else(|| dets[0])
}

/// Embed a legacy `Detection` through whichever pipeline is active.
/// Returns `None` when no landmarks can be matched (ArcFace) or the
/// forward pass collapses (EmbedNet).
#[allow(clippy::needless_pass_by_value)]
fn rank_for_det(
    embedder: &EmbedderKind,
    rgb: &RgbImage,
    gray: &GrayImage,
    det: &Detection,
    scrfd_dets: &[FaceDetection],
    rank_fn: impl FnOnce(&Embedding) -> Vec<(String, f32)>,
) -> Option<Vec<(String, f32)>> {
    let emb = embed_via_active(embedder, rgb, gray, det, scrfd_dets)?;
    Some(rank_fn(&emb))
}

/// `gray` lookup helper: pick the right crop source for legacy vs ArcFace.
fn detector_for_legacy(gray: &GrayImage) -> Vec<Detection> {
    if let Some(d) = build_haar_detector() {
        d.detect(gray)
    } else {
        Vec::new()
    }
}

/// Pick the first available 5-landmark face from a SCRFD batch.
fn pick_landmarks(scrfd_dets: &[FaceDetection]) -> Option<Landmarks> {
    let fd = pick_largest_face_detection(scrfd_dets);
    fd.landmarks
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
    // Placeholder slot for a person with no faces yet. The identity exists
    // (FK targets it from /api/jobs/{id}) but no probe can match it: any
    // probe at all will have positive cosine distance, so an attacker who
    // guesses the label gets nothing. The real embedding lands here on the
    // first successful `/api/persons/{id}/faces` enrollment.
    Embedding::from_raw(&[1e-6_f32; ARCFACE_DIM]).expect("constant vector")
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
