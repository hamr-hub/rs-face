//! ffmpeg pipe source — shell out to `ffmpeg` to decode arbitrary containers
//! into raw grayscale frames.
//!
//! Zero Rust deps: requires `ffmpeg` and (optionally) `ffprobe` on PATH. Uses
//! ffprobe to determine the source video dimensions, then asks ffmpeg to emit
//! raw gray video via stdout with `scale=480:-2`. The decoder runs on a
//! background thread and pushes `Frame` values into a bounded `mpsc` channel
//! so decode can overlap with detection.

use super::{Frame, FrameSource};
use crate::image::{GrayImage, RgbImage};
use std::io::{self, BufReader, Read};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;

pub struct FfmpegPipeSource {
    rx: Option<Receiver<Frame>>,
    decoder: Option<thread::JoinHandle<()>>,
    child: Option<Child>,
    fps: u32,
    width: usize,
    height: usize,
}

impl FfmpegPipeSource {
    pub fn new(url: &str, fps: u32) -> io::Result<Self> {
        // Probe source dimensions via ffprobe (best-effort).
        let (src_w, src_h) = probe_size(url).unwrap_or((640, 480));
        // Compute the actual output dimensions ffmpeg will produce with
        // `scale=480:-2`: ffmpeg rounds the result *up* to the nearest even
        // number to satisfy H.264/H.265 macroblock alignment, so we must
        // match that exactly or the decode loop will desync. The previous
        // implementation used `.round() & !1` which rounds half-to-even —
        // diverging from ffmpeg whenever the scaled height landed on .5.
        let raw_h = (src_h as f64 * 480.0 / src_w as f64);
        let h_even = (raw_h.ceil() as usize) & !1;
        let h_even = h_even.max(2);
        // Sanity: we also probe ffmpeg itself if available to confirm.
        let w = 480usize;
        let h = probe_even_height(url, w).unwrap_or(h_even);

        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner", "-loglevel", "error",
                "-i", url,
                "-an",
                "-vf", &format!("fps={},scale=480:-2,format=gray", fps),
                "-f", "rawvideo",
                "-pix_fmt", "gray",
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .stdin(Stdio::null())
            .spawn()?;
        let stdout = child.stdout.take()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "no stdout"))?;
        let stdout = BufReader::new(stdout);

        let (tx, rx) = mpsc::sync_channel::<Frame>(4);
        let decoder_fps = fps as u64;
        let decoder = thread::Builder::new()
            .name("rsface-ffmpeg-decoder".into())
            .spawn(move || {
                decode_loop(stdout, tx, w, h, decoder_fps);
            })
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        Ok(Self {
            rx: Some(rx),
            decoder: Some(decoder),
            child: Some(child),
            fps,
            width: w,
            height: h,
        })
    }
}

impl FrameSource for FfmpegPipeSource {
    fn next_frame(&mut self) -> io::Result<Option<Frame>> {
        let rx = match self.rx.as_ref() {
            Some(r) => r,
            None => return Ok(None),
        };
        match rx.recv() {
            Ok(f) => Ok(Some(f)),
            Err(_) => Ok(None),
        }
    }
}

impl Drop for FfmpegPipeSource {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        // Drop the channel first so the decoder thread's `send` returns Err and
        // it exits its loop promptly when the child is killed.
        drop(self.rx.take());
        if let Some(h) = self.decoder.take() {
            let _ = h.join();
        }
    }
}

/// Probe a media file for its video stream dimensions via `ffprobe`.
/// Returns `None` if ffprobe is missing or the file has no video stream.
fn probe_size(url: &str) -> Option<(usize, usize)> {
    let out = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            url,
        ])
        .output()
        .ok()?;
    if !out.status.success() { return None; }
    let s = std::str::from_utf8(&out.stdout).ok()?.trim();
    let mut parts = s.split(',');
    let w: usize = parts.next()?.trim().parse().ok()?;
    let h: usize = parts.next()?.trim().parse().ok()?;
    Some((w, h))
}

/// Ask ffmpeg itself what the even-aligned height will be for `scale=W:-2`.
/// This eliminates any drift between our rounded math and ffmpeg's own
/// rounding (which can differ by 2px on certain aspect ratios, e.g. 9:16).
fn probe_even_height(url: &str, w: usize) -> Option<usize> {
    let out = Command::new("ffmpeg")
        .args([
            "-hide_banner", "-loglevel", "error",
            "-i", url,
            "-an",
            "-vf", &format!("scale={}:-2", w),
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "-frames:v", "1",
            "-",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() { return None; }
    if out.stdout.is_empty() { return None; }
    let h = out.stdout.len() / w;
    if h == 0 { None } else { Some(h) }
}

/// Background decoder thread: read `w*h` bytes per frame, convert to GrayImage
/// + a grayscale-as-RGB RgbImage (so the sink can always draw boxes without
/// separate RGB extraction), and push into the channel.
fn decode_loop(mut stdout: BufReader<ChildStdout>, tx: mpsc::SyncSender<Frame>,
               w: usize, h: usize, fps: u64) {
    let frame_bytes = w * h;
    let mut buf = vec![0u8; frame_bytes];
    let mut frame_index: u64 = 0;
    loop {
        match stdout.read_exact(&mut buf) {
            Ok(()) => {
                let mut gray = GrayImage::new(w, h);
                gray.as_mut_slice().copy_from_slice(&buf);
                // Build a matching RGB by replicating gray into each channel.
                let mut rgb = RgbImage::new(w, h);
                for y in 0..h {
                    let grow = gray.row(y);
                    let rrow = rgb.row_mut(y);
                    for x in 0..w {
                        let v = grow[x];
                        rrow[x*3] = v;
                        rrow[x*3+1] = v;
                        rrow[x*3+2] = v;
                    }
                }
                let ts = if fps > 0 { frame_index * 1000 / fps } else { 0 };
                let frame = Frame {
                    index: frame_index,
                    timestamp_ms: ts,
                    gray: Arc::new(gray),
                    rgb: Some(Arc::new(rgb)),
                };
                frame_index += 1;
                if tx.send(frame).is_err() { break; }
            }
            Err(_) => break,
        }
    }
}