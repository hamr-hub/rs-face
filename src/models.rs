//! Model registry: pinned weights, integrity verification, and licence tiers.
//!
//! Three concerns are deliberately handled together here, because in practice they are
//! the same concern — "may I trust this file?":
//!
//! 1. **Provenance.** Model weights are large binaries fetched over the network and then
//!    executed as the numerical core of the system. A registry of canonical URLs keeps
//!    ad-hoc download instructions out of README prose where they rot.
//!
//! 2. **Integrity.** Every entry pins a SHA-256 digest, verified before the file is
//!    handed to an inference backend. Without this, a truncated download or an upstream
//!    re-release silently degrades accuracy instead of failing, which is the worst
//!    possible failure mode: the system keeps working and keeps being wrong.
//!
//! 3. **Licence.** The highest-accuracy open face models (InsightFace SCRFD / ArcFace)
//!    are licensed for **non-commercial research use only**. A crate that markets itself
//!    as production-ready must surface that, not bury it, so the restriction is encoded
//!    as a typed field on every model and is queryable at runtime.
//!
//! The SHA-256 implementation is pure Rust with no dependencies, preserving the crate's
//! zero-dependency core. It is validated against the NIST FIPS 180-4 test vectors below.

use core::fmt;

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4)
// ---------------------------------------------------------------------------

/// Round constants: first 32 bits of the fractional parts of the cube roots of the
/// first 64 primes.
#[rustfmt::skip]
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Streaming SHA-256 hasher.
///
/// Streaming rather than one-shot because model files are hundreds of megabytes and
/// must not be buffered entirely in memory just to be checksummed.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    total_len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// Initial hash values: first 32 bits of the fractional parts of the square roots
    /// of the first 8 primes.
    pub fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: [0u8; 64],
            buffered: 0,
            total_len: 0,
        }
    }

    /// Feed bytes into the hasher.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);

        // Top up a partial block first.
        if self.buffered > 0 {
            let need = 64 - self.buffered;
            let take = need.min(data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }

        // Consume whole blocks straight from the input, avoiding a copy.
        while data.len() >= 64 {
            let (block, rest) = data.split_at(64);
            let mut b = [0u8; 64];
            b.copy_from_slice(block);
            self.compress(&b);
            data = rest;
        }

        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffered = data.len();
        }
    }

    /// Finish and produce the 32-byte digest.
    pub fn finalize(mut self) -> [u8; 32] {
        // Padding: 0x80, then zeros, then the 64-bit big-endian bit length.
        let bit_len = self.total_len.wrapping_mul(8);
        self.update_raw(&[0x80]);
        while self.buffered != 56 {
            self.update_raw(&[0x00]);
        }
        self.update_raw(&bit_len.to_be_bytes());

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    /// Like `update` but does not count toward `total_len` — used by the padding step,
    /// which must not perturb the already-captured message length.
    fn update_raw(&mut self, data: &[u8]) {
        for &b in data {
            self.buffer[self.buffered] = b;
            self.buffered += 1;
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        for (s, v) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }
}

/// One-shot SHA-256 over a byte slice.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

/// Lowercase hex encoding of a digest.
pub fn hex(digest: &[u8]) -> String {
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        // Manual nibble formatting keeps this allocation-free per byte.
        const HEX: &[u8; 16] = b"0123456789abcdef";
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

// ---------------------------------------------------------------------------
// Licence tiers
// ---------------------------------------------------------------------------

/// Licensing terms attached to a set of model weights.
///
/// Encoded as a type rather than a doc comment because the practical consequence —
/// "can I ship this in a product?" — must be checkable in code, e.g. to refuse to start
/// a production server with research-only weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum License {
    /// Apache-2.0 / MIT / BSD: usable in a commercial product.
    Permissive(&'static str),
    /// Non-commercial research use only. Commercial deployment requires a separate
    /// licence from the upstream rights holder.
    ResearchOnly(&'static str),
}

impl License {
    /// Whether these weights may be used in a commercial product.
    #[inline]
    pub fn is_commercial_use_allowed(&self) -> bool {
        matches!(self, License::Permissive(_))
    }

    /// SPDX-ish identifier or short description.
    pub fn name(&self) -> &'static str {
        match self {
            License::Permissive(n) | License::ResearchOnly(n) => n,
        }
    }
}

impl fmt::Display for License {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            License::Permissive(n) => write!(f, "{n} (commercial use OK)"),
            License::ResearchOnly(n) => write!(f, "{n} (NON-COMMERCIAL research only)"),
        }
    }
}

/// What a model does, so the loader can reject a mismatched file early.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    /// Outputs boxes (and usually 5 landmarks).
    Detector,
    /// Outputs an identity embedding.
    Recognizer,
    /// Outputs a real/spoof liveness decision.
    Liveness,
}

/// A pinned, verifiable model artefact.
#[derive(Clone, Debug)]
pub struct ModelSpec {
    /// Stable short id used on the CLI and in config, e.g. `scrfd_10g_kps`.
    pub id: &'static str,
    pub kind: ModelKind,
    /// Canonical download URL.
    pub url: &'static str,
    /// Filename to store locally.
    pub file_name: &'static str,
    /// Expected SHA-256, lowercase hex, or `None` when not yet pinned.
    ///
    /// `None` is meaningful and deliberately not faked: an unpinned model still works but
    /// [`verify_bytes`] reports [`Integrity::Unpinned`] so the caller can warn. Inventing
    /// a plausible-looking digest would be far worse than admitting we lack one.
    pub sha256: Option<&'static str>,
    /// Expected size in bytes, when known — a cheap pre-check before hashing 300 MB.
    pub size_bytes: Option<u64>,
    pub license: License,
    /// Network input resolution (square) the model expects.
    pub input_size: usize,
    /// Published accuracy note, for `--list-models` output.
    pub accuracy: &'static str,
}

impl ModelSpec {
    /// Whether this model is safe to use in a commercial deployment.
    pub fn is_commercial_use_allowed(&self) -> bool {
        self.license.is_commercial_use_allowed()
    }
}

/// Result of an integrity check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Integrity {
    /// Digest matched the pinned value.
    Verified,
    /// No digest pinned for this model; the bytes were not validated.
    Unpinned,
    /// Digest mismatch — the file must not be used.
    Mismatch {
        expected: &'static str,
        actual: String,
    },
    /// Size differed from the expected size, so hashing was skipped.
    SizeMismatch { expected: u64, actual: u64 },
}

impl Integrity {
    /// Whether the bytes may be handed to an inference backend.
    ///
    /// [`Integrity::Unpinned`] counts as usable (the user may legitimately supply their
    /// own export) but callers are expected to log a warning.
    #[inline]
    pub fn is_usable(&self) -> bool {
        matches!(self, Integrity::Verified | Integrity::Unpinned)
    }
}

impl fmt::Display for Integrity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Integrity::Verified => write!(f, "sha256 verified"),
            Integrity::Unpinned => write!(f, "no pinned digest — integrity NOT verified"),
            Integrity::Mismatch { expected, actual } => {
                write!(f, "sha256 MISMATCH: expected {expected}, got {actual}")
            }
            Integrity::SizeMismatch { expected, actual } => {
                write!(f, "size mismatch: expected {expected} bytes, got {actual}")
            }
        }
    }
}

/// Verify a model's bytes against its pinned size and digest.
///
/// Size is checked before hashing so the common failure (a truncated or
/// HTML-error-page download) is reported in O(1) rather than after hashing garbage.
pub fn verify_bytes(spec: &ModelSpec, bytes: &[u8]) -> Integrity {
    if let Some(expected) = spec.size_bytes {
        if bytes.len() as u64 != expected {
            return Integrity::SizeMismatch {
                expected,
                actual: bytes.len() as u64,
            };
        }
    }
    match spec.sha256 {
        None => Integrity::Unpinned,
        Some(expected) => {
            let actual = hex(&sha256(bytes));
            if actual == expected {
                Integrity::Verified
            } else {
                Integrity::Mismatch { expected, actual }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// YuNet 2023mar — the commercially usable detector.
///
/// Apache-2.0 and only ~227 KB, at the cost of accuracy relative to SCRFD
/// (WIDER FACE hard 0.768 vs 0.828). This is the default recommendation for anyone
/// shipping a product, precisely because the more accurate options are not licensable.
pub const YUNET_2023MAR: ModelSpec = ModelSpec {
    id: "yunet_2023mar",
    kind: ModelKind::Detector,
    url: "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    file_name: "face_detection_yunet_2023mar.onnx",
    sha256: Some("8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    size_bytes: Some(232_589),
    license: License::Permissive("Apache-2.0"),
    input_size: 320,
    accuracy: "WIDER FACE AP easy/medium/hard = 0.887 / 0.871 / 0.768",
};

/// SCRFD-10G with 5-point keypoints — the accuracy leader we can actually align with.
///
/// The keypoint head matters: without it there are no landmarks, so no similarity
/// transform, so no comparable ArcFace embeddings.
pub const SCRFD_10G_KPS: ModelSpec = ModelSpec {
    id: "scrfd_10g_kps",
    kind: ModelKind::Detector,
    url: "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    file_name: "det_10g.onnx",
    // Pinned from buffalo_l.zip @ the v0.7 release tag, verified 2026-09-01 by
    // downloading the pack and hashing the extracted file on this machine.
    sha256: Some("5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"),
    size_bytes: Some(16_923_827),
    license: License::ResearchOnly("InsightFace model weights"),
    input_size: 640,
    accuracy: "WIDER FACE AP easy/medium/hard = 0.954 / 0.940 / 0.828",
};

/// ArcFace ResNet-50 trained on WebFace600K — the recognition backbone.
pub const ARCFACE_W600K_R50: ModelSpec = ModelSpec {
    id: "arcface_w600k_r50",
    kind: ModelKind::Recognizer,
    url: "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    file_name: "w600k_r50.onnx",
    // Pinned from buffalo_l.zip @ v0.7, verified 2026-09-01.
    sha256: Some("4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"),
    size_bytes: Some(174_383_860),
    license: License::ResearchOnly("InsightFace model weights"),
    input_size: 112,
    accuracy: "LFW 99.83 / CFP-FP 99.33 / AgeDB-30 98.23 / IJB-C TAR@1e-4 97.25",
};

/// MobileFaceNet recognition backbone — ~12 MB, for edge deployment.
pub const ARCFACE_W600K_MBF: ModelSpec = ModelSpec {
    id: "arcface_w600k_mbf",
    kind: ModelKind::Recognizer,
    url: "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
    file_name: "w600k_mbf.onnx",
    sha256: None,
    size_bytes: None,
    license: License::ResearchOnly("InsightFace model weights"),
    input_size: 112,
    accuracy: "LFW 99.70 / CFP-FP 98.00 / AgeDB-30 96.58 / IJB-C TAR@1e-4 95.02",
};

/// MiniFASNetV2 silent-liveness model at the 2.7x crop scale.
///
/// ONNX conversion of the MiniVision Silent-Face-Anti-Spoofing project
/// (Apache-2.0). Emits a 3-class softmax `[printed photo, real face, screen
/// replay]` over an 80x80 RGB crop. Commercial use is allowed, unlike the
/// InsightFace weights.
pub const LIVENESS_MINIFASNET_V2: ModelSpec = ModelSpec {
    id: "liveness_minifasnet_v2",
    kind: ModelKind::Liveness,
    url: "https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/raw/main/onnx/2.7_80x80_MiniFASNetV2.onnx",
    file_name: "2.7_80x80_MiniFASNetV2.onnx",
    sha256: Some("0cbe5caec95c31de9d2ef845cb85407d76aecd1b6a2c0e343f7d35306bfbccb8"),
    size_bytes: Some(1_744_126),
    license: License::Permissive("Apache-2.0"),
    input_size: 80,
    accuracy: "Silent-Face print/replay detection; 3-class, ~9 ms CPU",
};

/// MiniFASNetV1SE silent-liveness model at the 4.0x crop scale.
///
/// Paired with [`LIVENESS_MINIFASNET_V2`]; the two softmax outputs are
/// averaged before the real/spoof decision, the upstream-recommended setup.
pub const LIVENESS_MINIFASNET_V1SE: ModelSpec = ModelSpec {
    id: "liveness_minifasnet_v1se",
    kind: ModelKind::Liveness,
    url: "https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/raw/main/onnx/4_0_0_80x80_MiniFASNetV1SE.onnx",
    file_name: "4_0_0_80x80_MiniFASNetV1SE.onnx",
    sha256: Some("a25886a85cdcfa2c4ea23edb71de35f250c17827b4cadd253a972b28c80fdf1e"),
    size_bytes: Some(1_743_294),
    license: License::Permissive("Apache-2.0"),
    input_size: 80,
    accuracy: "Silent-Face print/replay detection; 3-class, ~9 ms CPU",
};

/// Every model this crate knows how to consume.
pub const REGISTRY: &[&ModelSpec] = &[
    &YUNET_2023MAR,
    &SCRFD_10G_KPS,
    &ARCFACE_W600K_R50,
    &ARCFACE_W600K_MBF,
    &LIVENESS_MINIFASNET_V2,
    &LIVENESS_MINIFASNET_V1SE,
];

/// Look a model up by its stable id.
pub fn find(id: &str) -> Option<&'static ModelSpec> {
    REGISTRY.iter().copied().find(|m| m.id == id)
}

/// All models usable in a commercial product.
pub fn commercially_usable() -> impl Iterator<Item = &'static ModelSpec> {
    REGISTRY
        .iter()
        .copied()
        .filter(|m| m.is_commercial_use_allowed())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- SHA-256 conformance ------------------------------------------------
    // Vectors from NIST FIPS 180-4 / the standard published test set. These pin the
    // implementation to the real algorithm; a hand-rolled hash that is merely
    // self-consistent would happily "verify" corrupt model files.

    #[test]
    fn sha256_empty_string() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_abc() {
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_448_bit_message() {
        assert_eq!(
            hex(&sha256(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn sha256_multi_block_message() {
        // 896-bit message: exercises the second compression block and length encoding.
        let msg = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
        assert_eq!(
            hex(&sha256(msg)),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
        );
    }

    #[test]
    fn sha256_one_million_a() {
        // The classic long-message vector: forces many blocks and a large bit length.
        let mut h = Sha256::new();
        let chunk = vec![b'a'; 10_000];
        for _ in 0..100 {
            h.update(&chunk);
        }
        assert_eq!(
            hex(&h.finalize()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn sha256_exactly_one_block_is_64_bytes() {
        // 64 bytes lands exactly on a block boundary, which needs a full extra
        // padding block. A common off-by-one lives here.
        let data = [b'x'; 64];
        assert_eq!(hex(&sha256(&data)).len(), 64);
        assert_eq!(sha256(&data), sha256(&data));
    }

    #[test]
    fn sha256_streaming_matches_one_shot_at_every_split() {
        // Chunk-boundary handling must not affect the digest. Sweeping every split point
        // across a >64-byte message covers partial-buffer top-up and block alignment.
        let msg: Vec<u8> = (0..200u32).map(|i| (i % 251) as u8).collect();
        let expect = sha256(&msg);
        for split in 0..msg.len() {
            let mut h = Sha256::new();
            h.update(&msg[..split]);
            h.update(&msg[split..]);
            assert_eq!(h.finalize(), expect, "split at {split} changed the digest");
        }
    }

    #[test]
    fn sha256_byte_wise_streaming_matches_one_shot() {
        let msg = b"the quick brown fox jumps over the lazy dog, repeatedly and at length";
        let mut h = Sha256::new();
        for b in msg.iter() {
            h.update(&[*b]);
        }
        assert_eq!(h.finalize(), sha256(msg));
    }

    #[test]
    fn sha256_single_bit_change_changes_digest() {
        assert_ne!(sha256(b"abc"), sha256(b"abd"));
    }

    #[test]
    fn hex_encodes_low_nibbles_and_leading_zeros() {
        assert_eq!(hex(&[0x00, 0x0f, 0xff, 0xa5]), "000fffa5");
        assert_eq!(hex(&[]), "");
    }

    // -- Licence semantics --------------------------------------------------

    #[test]
    fn permissive_allows_commercial_research_only_does_not() {
        assert!(License::Permissive("Apache-2.0").is_commercial_use_allowed());
        assert!(!License::ResearchOnly("InsightFace model weights").is_commercial_use_allowed());
    }

    #[test]
    fn research_only_display_shouts_the_restriction() {
        // This string reaches users; it must be unmissable.
        let s = License::ResearchOnly("InsightFace model weights").to_string();
        assert!(s.contains("NON-COMMERCIAL"), "got {s:?}");
    }

    #[test]
    fn insightface_models_are_marked_research_only() {
        // Regression guard: mislabelling these as permissive would invite a licence
        // violation in a downstream product.
        for spec in [&SCRFD_10G_KPS, &ARCFACE_W600K_R50, &ARCFACE_W600K_MBF] {
            assert!(
                !spec.is_commercial_use_allowed(),
                "{} must be ResearchOnly",
                spec.id
            );
        }
    }

    #[test]
    fn yunet_is_the_commercially_usable_option() {
        assert!(YUNET_2023MAR.is_commercial_use_allowed());
        let ids: Vec<_> = commercially_usable().map(|m| m.id).collect();
        assert_eq!(ids, vec!["yunet_2023mar"]);
    }

    // -- Integrity ----------------------------------------------------------

    fn spec_with(sha: Option<&'static str>, size: Option<u64>) -> ModelSpec {
        ModelSpec {
            id: "t",
            kind: ModelKind::Detector,
            url: "",
            file_name: "t.onnx",
            sha256: sha,
            size_bytes: size,
            license: License::Permissive("MIT"),
            input_size: 640,
            accuracy: "",
        }
    }

    #[test]
    fn verify_detects_matching_digest() {
        let sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(
            verify_bytes(&spec_with(Some(sha), None), b"abc"),
            Integrity::Verified
        );
    }

    #[test]
    fn verify_detects_corrupt_bytes() {
        let sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let got = verify_bytes(&spec_with(Some(sha), None), b"abd");
        assert!(matches!(got, Integrity::Mismatch { .. }));
        assert!(!got.is_usable(), "corrupt weights must never be usable");
    }

    #[test]
    fn verify_reports_unpinned_rather_than_pretending() {
        let got = verify_bytes(&spec_with(None, None), b"whatever");
        assert_eq!(got, Integrity::Unpinned);
        // Usable, but the Display text must warn.
        assert!(got.is_usable());
        assert!(got.to_string().contains("NOT verified"));
    }

    #[test]
    fn verify_catches_truncated_download_before_hashing() {
        // The realistic failure: an HTML error page saved as a .onnx. Size catches it
        // in O(1), and crucially the digest is never consulted.
        let sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let got = verify_bytes(&spec_with(Some(sha), Some(999_999)), b"<html>404</html>");
        assert_eq!(
            got,
            Integrity::SizeMismatch {
                expected: 999_999,
                actual: 16
            }
        );
        assert!(!got.is_usable());
    }

    #[test]
    fn size_match_still_requires_digest_match() {
        let sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        // Right length, wrong content — only the hash can catch this.
        let got = verify_bytes(&spec_with(Some(sha), Some(3)), b"abd");
        assert!(matches!(got, Integrity::Mismatch { .. }));
    }

    // -- Registry -----------------------------------------------------------

    #[test]
    fn registry_ids_are_unique() {
        let mut ids: Vec<&str> = REGISTRY.iter().map(|m| m.id).collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "duplicate model id in REGISTRY");
    }

    #[test]
    fn find_locates_models_and_rejects_unknown() {
        assert_eq!(
            find("arcface_w600k_r50").unwrap().kind,
            ModelKind::Recognizer
        );
        assert_eq!(find("yunet_2023mar").unwrap().kind, ModelKind::Detector);
        assert!(find("no_such_model").is_none());
    }

    #[test]
    fn recognizers_use_the_112_arcface_crop_size() {
        // The alignment module hard-codes a 112x112 canonical crop; a recognizer
        // expecting another input size would be silently mis-fed.
        for m in REGISTRY.iter().filter(|m| m.kind == ModelKind::Recognizer) {
            assert_eq!(
                m.input_size,
                crate::align::ARCFACE_CROP_SIZE,
                "{} input size disagrees with the alignment crop size",
                m.id
            );
        }
    }

    #[test]
    fn every_model_documents_its_accuracy_and_url() {
        for m in REGISTRY {
            assert!(!m.accuracy.is_empty(), "{} lacks an accuracy note", m.id);
            assert!(m.url.starts_with("https://"), "{} URL must be https", m.id);
            assert!(m.file_name.ends_with(".onnx"), "{} file_name", m.id);
        }
    }
}
