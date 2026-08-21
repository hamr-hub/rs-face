//! Multi-threaded detection pipeline.
//!
//! Layout (single source, N detector workers, single sink):
//!
//! ```text
//!   [source thread] --frame--> [dispatch thread] --round-robin-->
//!     worker[0] | worker[1] | ... | worker[N-1]
//!         \         \              /
//!          --------> [result channel] --------> [sink thread]
//! ```
//!
//! Backpressure: each worker channel is bounded by a small capacity; the
//! source blocks when all are full, naturally throttling the producer.

use crate::detector::{Detection, Detector, DetectorConfig};
use crate::haar::Cascade;
use crate::image::GrayImage;
use crate::output::{write_annotated_png, write_manifest, DetectionRecord, PipelineSummary};
use crate::source::{Frame, FrameSource};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub threads: usize,
    pub queue_depth: usize,
    pub detector: DetectorConfig,
    pub min_score: f32,
    /// Only emit frames that contain at least one detection.
    pub only_with_face: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            threads: std::thread::available_parallelism()
                .map(|n| n.get().max(1))
                .unwrap_or(1),
            queue_depth: 4,
            detector: DetectorConfig::default(),
            min_score: 0.0,
            only_with_face: false,
        }
    }
}

pub struct PipelineStats {
    pub frames_processed: u64,
    pub frames_with_face: u64,
    pub total_detections: u64,
    pub elapsed_ms: u64,
    /// Mean per-frame pure-detection time in ms (excludes I/O and PNG
    /// encoding); 0.0 when unavailable.
    pub detect_ms_avg: f64,
}

#[derive(Debug, Clone)]
struct WorkItem {
    frame: Frame,
    seq: u64,
}

struct WorkResult {
    seq: u64,
    frame: Frame,
    detections: Vec<Detection>,
    /// Per-frame detection time in microseconds (surfaced in the manifest
    /// as `detect_ms` and aggregated into `PipelineStats::detect_ms_avg`).
    elapsed_us: u64,
}

pub struct Pipeline;

impl Pipeline {
    /// Run the full pipeline: read frames from `source`, detect with `cascade`,
    /// write annotated PNGs and a manifest to `output_dir`.
    pub fn run(
        source: &mut dyn FrameSource,
        cascade: Cascade,
        output_dir: &Path,
        cfg: PipelineConfig,
    ) -> std::io::Result<PipelineStats> {
        let start = Instant::now();
        std::fs::create_dir_all(output_dir)?;
        let detector = Arc::new(Detector::new(cascade, cfg.detector.clone()));

        // Per-worker bounded channels (source → workers).
        let n = cfg.threads.max(1);
        let cap = cfg.queue_depth.max(1);
        let mut worker_txs: Vec<mpsc::SyncSender<Option<WorkItem>>> = Vec::with_capacity(n);
        let mut worker_rxs: Vec<Receiver<Option<WorkItem>>> = Vec::with_capacity(n);
        for _ in 0..n {
            let (tx, rx) = mpsc::sync_channel::<Option<WorkItem>>(cap);
            worker_txs.push(tx);
            worker_rxs.push(rx);
        }
        let (result_tx, result_rx) = mpsc::channel::<WorkResult>();

        let cancel = Arc::new(AtomicBool::new(false));
        let _frames_seen = Arc::new(AtomicU64::new(0));
        let mut workers = Vec::with_capacity(n);
        for i in 0..n {
            let rx = worker_rxs.pop().expect("worker_rxs drained in same order");
            let det = detector.clone();
            let result_tx = result_tx.clone();
            let cancel = cancel.clone();
            let min_score = cfg.min_score;
            let h = thread::Builder::new()
                .name(format!("rsface-det-{i}"))
                .spawn(move || {
                    worker_loop(rx, det, min_score, result_tx, cancel);
                })?;
            workers.push(h);
        }
        drop(result_tx); // workers hold the only senders

        // Sink thread.
        let only_with_face = cfg.only_with_face;
        let sink_output_dir = output_dir.to_path_buf();
        let sink_handle = thread::Builder::new()
            .name("rsface-sink".to_string())
            .spawn(move || -> std::io::Result<Vec<DetectionRecord>> {
                sink_loop(result_rx, &sink_output_dir, only_with_face)
            })?;

        // Dispatcher thread: round-robin frames from source to workers.
        // Run the dispatcher on the calling thread to keep the API simple.
        let mut next_worker = 0usize;
        let mut seq: u64 = 0;
        loop {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let frame_opt = source.next_frame()?;
            let Some(frame) = frame_opt else {
                break;
            };
            _frames_seen.fetch_add(1, Ordering::Relaxed);
            let item = WorkItem { frame, seq };
            // Round-robin: try the next worker; if its queue is full, fall back to any non-full channel.
            let mut placed = false;
            for k in 0..n {
                let idx = (next_worker + k) % n;
                if worker_txs[idx].try_send(Some(item.clone())).is_ok() {
                    next_worker = (idx + 1) % n;
                    placed = true;
                    break;
                }
            }
            if !placed {
                // All queues full → backpressure: block on the next worker.
                if let Err(e) = worker_txs[next_worker].send(Some(item)) {
                    if cancel.load(Ordering::Relaxed) {
                        break;
                    }
                    return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, e));
                }
                next_worker = (next_worker + 1) % n;
            }
            seq += 1;
        }
        // Send None to all workers to signal EOF.
        for tx in &worker_txs {
            let _ = tx.send(None);
        }
        drop(worker_txs);

        // Wait for workers + sink.
        for h in workers {
            let _ = h.join();
        }
        let records = sink_handle
            .join()
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "sink panicked"))??;

        let summary = PipelineSummary {
            frames_processed: records.len() as u64,
            frames_with_face: records.iter().filter(|r| !r.detections.is_empty()).count() as u64,
            total_detections: records.iter().map(|r| r.detections.len() as u64).sum(),
            elapsed_ms: start.elapsed().as_millis() as u64,
            detect_ms_total: records.iter().map(|r| r.detect_ms).sum(),
        };
        let manifest = output_dir.join("manifest.json");
        write_manifest(&manifest, &records, &summary)?;
        Ok(PipelineStats {
            frames_processed: summary.frames_processed,
            frames_with_face: summary.frames_with_face,
            total_detections: summary.total_detections,
            elapsed_ms: summary.elapsed_ms,
            detect_ms_avg: summary.detect_ms_avg(),
        })
    }
}

fn worker_loop(
    rx: Receiver<Option<WorkItem>>,
    detector: Arc<Detector>,
    min_score: f32,
    result_tx: Sender<WorkResult>,
    cancel: Arc<AtomicBool>,
) {
    while let Ok(Some(item)) = rx.recv() {
        let start = Instant::now();
        let detections = detector.detect(&item.frame.gray);
        let elapsed_us = start.elapsed().as_micros() as u64;
        let detections: Vec<Detection> = detections
            .into_iter()
            .filter(|d| d.score >= min_score)
            .collect();
        let res = WorkResult {
            seq: item.seq,
            frame: item.frame,
            detections,
            elapsed_us,
        };
        if result_tx.send(res).is_err() {
            cancel.store(true, Ordering::Relaxed);
            break;
        }
    }
}

fn sink_loop(
    rx: Receiver<WorkResult>,
    output_dir: &Path,
    only_with_face: bool,
) -> std::io::Result<Vec<DetectionRecord>> {
    let mut records = Vec::new();
    // Sink must receive results in arrival order; we don't reorder because we want
    // manifest order to reflect detection order, not source order.
    let mut seq_map: std::collections::BTreeMap<u64, WorkResult> =
        std::collections::BTreeMap::new();
    let mut next_seq: u64 = 0;
    while let Ok(res) = rx.recv() {
        seq_map.insert(res.seq, res);
        while let Some(r) = seq_map.remove(&next_seq) {
            next_seq += 1;
            if only_with_face && r.detections.is_empty() {
                continue;
            }
            let rgb = r
                .frame
                .rgb
                .clone()
                .map(|a| (*a).clone())
                .unwrap_or_else(|| {
                    // Reconstruct an RGB from gray by replication so we can always draw boxes.
                    let g = &r.frame.gray;
                    let mut rgb = crate::image::RgbImage::new(g.width(), g.height());
                    for y in 0..g.height() {
                        for x in 0..g.width() {
                            let v = (*g)[(x, y)];
                            let row = rgb.row_mut(y);
                            row[x * 3] = v;
                            row[x * 3 + 1] = v;
                            row[x * 3 + 2] = v;
                        }
                    }
                    rgb
                });
            let rec = DetectionRecord {
                frame_index: r.frame.index,
                timestamp_ms: r.frame.timestamp_ms,
                image_file: String::new(), // filled below
                width: rgb.width(),
                height: rgb.height(),
                detections: r.detections.clone(),
                detect_ms: r.elapsed_us as f64 / 1000.0,
            };
            let fname = write_annotated_png(output_dir, &rec, &rgb)?;
            let mut rec = rec;
            rec.image_file = fname;
            records.push(rec);
        }
    }
    Ok(records)
}

#[allow(dead_code)]
fn _ensure_grayimage_send(_: &GrayImage) {}
