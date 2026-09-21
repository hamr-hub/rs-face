//! Multi-recogniser consensus support.
//!
//! Loads a registered face gallery from a directory of identity folders,
//! builds the zero-dependency recognisers (LBPH, eigenfaces, Fisherfaces)
//! on it, and identifies every face in a probe image, fusing the per-
//! recogniser outcomes through
//! [`rsface::ensemble::fuse_recognitions`].

use crate::jobs::DetectorKind;
use rsface::eigenface::{EigenfaceConfig, EigenfaceRecognizer};
use rsface::face::Detection;
use rsface::image::codec::{read_pgm, read_ppm};
use rsface::image::png::decode_to_gray;
use rsface::image::GrayImage;
use rsface::lbph::{LbphConfig, LbphRecognizer};
use rsface::fisherface::{FisherfaceConfig, FisherfaceRecognizer};
use rsface::recognizer::{FaceRecognizer, Recognition};
use rsface::ensemble::{fuse_recognitions, RecognitionFusionConfig, TaggedRecognition};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// One decoded identity crop.
type LabelledCrop = (String, GrayImage);

/// Trained recogniser plus its accept threshold. `+ Send` 让 `Box<dyn …>`
/// 可在 spawn_blocking / Arc 共享 — `FaceRecognizer` trait 已要求 `Send`
/// (see `recognizer.rs`),这里再加一遍 trait-object 边界让编译器检查到。
pub struct RecognizerEntry {
    pub recognizer: Box<dyn FaceRecognizer + Send + Sync>,
    pub threshold: f32,
}

/// A gallery with every recogniser that trained successfully on it.
///
/// `recognizers` 必须 `pub`:handler 要在空画廊场景返 `no_gallery`
/// (区分于 `GalleryBundle::load` 失败的 IO 错)。其它字段给前端响应 JSON 用。
pub struct GalleryBundle {
    pub recognizers: Vec<RecognizerEntry>,
    pub identities: usize,
    pub crops: usize,
}

/// Load one image file into a [`GrayImage`], using the core codecs.
fn load_crop(path: &Path) -> Option<GrayImage> {
    let bytes = fs::read(path).ok()?;
    let mut cur = Cursor::new(bytes);
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "pgm" => read_pgm(&mut cur).ok(),
        "ppm" => read_ppm(&mut cur).ok().map(|img| img.to_gray()),
        "png" => decode_to_gray(&mut cur).ok(),
        _ => None,
    }
}

/// Read every identity folder under `dir` (one subdirectory per label).
fn load_gallery_crops(dir: &Path) -> std::io::Result<Vec<LabelledCrop>> {
    let mut folders: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    folders.sort();

    let mut crops = Vec::new();
    for folder in folders {
        let label = folder
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default();
        let mut files: Vec<PathBuf> = fs::read_dir(&folder)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.is_file())
            .collect();
        files.sort();
        for file in files {
            if let Some(img) = load_crop(&file) {
                crops.push((label.clone(), img));
            }
        }
    }
    Ok(crops)
}

impl GalleryBundle {
    /// Build the gallery and train every recogniser on it. Recognisers
    /// whose assumptions the gallery violates (e.g. Fisherfaces needs one
    /// identity enrolled twice) are skipped rather than failing the load.
    pub fn load(dir: &Path) -> std::io::Result<Self> {
        let crops = load_gallery_crops(dir)?;
        let identities = crops
            .iter()
            .map(|(label, _)| label.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .len();

        let mut recognizers: Vec<RecognizerEntry> = Vec::new();

        // LBPH: incremental enrolment, always trains.
        let lbph_config = LbphConfig::default();
        let mut lbph = LbphRecognizer::new(lbph_config);
        for (label, img) in &crops {
            lbph.enroll(label.clone(), img);
        }
        recognizers.push(RecognizerEntry {
            threshold: lbph_config.max_distance,
            recognizer: Box::new(lbph),
        });

        // Eigenfaces: needs at least two crops.
        let eigen_samples: Vec<(&str, &GrayImage)> =
            crops.iter().map(|(label, img)| (label.as_str(), img)).collect();
        let eigen_config = EigenfaceConfig::default();
        if let Ok(recognizer) = EigenfaceRecognizer::train(eigen_config, eigen_samples) {
            recognizers.push(RecognizerEntry {
                threshold: eigen_config.max_distance,
                recognizer: Box::new(recognizer),
            });
        }

        // Fisherfaces: needs at least two classes and one repeat crop.
        let fisher_samples: Vec<(&str, &GrayImage)> =
            crops.iter().map(|(label, img)| (label.as_str(), img)).collect();
        let fisher_config = FisherfaceConfig::default();
        if let Ok(recognizer) = FisherfaceRecognizer::train(fisher_config, fisher_samples)
        {
            recognizers.push(RecognizerEntry {
                threshold: fisher_config.max_distance,
                recognizer: Box::new(recognizer),
            });
        }

        Ok(Self {
            recognizers,
            identities,
            crops: crops.len(),
        })
    }
}

/// Crop the grayscale region described by `detection` from `base`.
fn crop_gray(base: &GrayImage, detection: &Detection) -> GrayImage {
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

/// How one recogniser voted, for the per-recogniser detail view.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RecognizerVoteDetail {
    pub source: String,
    pub status: &'static str,
    pub label: Option<String>,
    pub distance: Option<f32>,
}

fn vote_detail(recognition: &Recognition, source: &str) -> RecognizerVoteDetail {
    match recognition {
        Recognition::Match { label, distance, .. } => RecognizerVoteDetail {
            source: source.to_string(),
            status: "match",
            label: Some(label.clone()),
            distance: Some(*distance),
        },
        Recognition::BelowThreshold { best } => RecognizerVoteDetail {
            source: source.to_string(),
            status: "below_threshold",
            label: best.as_ref().map(|(label, _)| label.clone()),
            distance: best.as_ref().map(|(_, distance)| *distance),
        },
        Recognition::Ambiguous { first, .. } => RecognizerVoteDetail {
            source: source.to_string(),
            status: "ambiguous",
            label: Some(first.clone()),
            distance: None,
        },
        Recognition::NoCandidates => RecognizerVoteDetail {
            source: source.to_string(),
            status: "no_candidates",
            label: None,
            distance: None,
        },
    }
}

/// The consensus identity for one detected face.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ConsensusIdentity {
    pub label: String,
    pub votes: usize,
    pub confidence: f32,
    pub sources: String,
}

/// One detected face with its multi-recogniser consensus.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RecognizedFace {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub score: f32,
    pub consensus: Option<ConsensusIdentity>,
    pub recognizers: Vec<RecognizerVoteDetail>,
}

/// Detect every face in `probe` and identify each one by multi-recogniser
/// consensus. `detector` supplies the face boxes; the gallery recognisers
/// supply the identity votes.
pub fn recognize_faces(
    detector: &DetectorKind,
    gallery: &GalleryBundle,
    probe: &GrayImage,
) -> Vec<RecognizedFace> {
    let detections = detector.detect(probe);
    let fusion_config = RecognitionFusionConfig::default();

    detections
        .iter()
        .map(|detection| {
            let face_crop = crop_gray(probe, detection);

            // Run every recogniser on this crop, then fuse in one pass.
            // Allocation: 1 × TaggedRecognition per recognizer (a 4-field
            // struct, ~32 bytes) — replaces the old 2 × Vec<(Recognition,
            // &str, f32)> intermediate that allocated a heap tuple per
            // recognizer per face.
            let mut tagged: Vec<TaggedRecognition> = Vec::with_capacity(gallery.recognizers.len());
            let mut recognizers: Vec<RecognizerVoteDetail> = Vec::with_capacity(gallery.recognizers.len());
            for entry in &gallery.recognizers {
                let recognition = entry.recognizer.identify_crop(&face_crop);
                let source = entry.recognizer.name();
                recognizers.push(vote_detail(&recognition, source));
                tagged.push(TaggedRecognition::new(
                    recognition,
                    source,
                    1.0,
                    entry.threshold,
                ));
            }
            let fused = fuse_recognitions(tagged, &fusion_config);

            RecognizedFace {
                x: detection.x,
                y: detection.y,
                w: detection.w,
                h: detection.h,
                score: detection.score,
                consensus: fused.first().map(|f| ConsensusIdentity {
                    label: f.label.clone(),
                    votes: f.votes,
                    confidence: f.confidence,
                    sources: f.sources.clone(),
                }),
                recognizers,
            }
        })
        .collect()
}
