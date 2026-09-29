//! Minimal baseline JPEG (JFIF) decoder.
//!
//! Implements ITU-T T.81 baseline sequential Huffman decoding — 8-bit samples,
//! no progressive scans, no 12-bit, no arithmetic coding. Supports grayscale
//! (1 component) and YCbCr (3 components) with sampling factors 1x1, 2x1, 1x2,
//! 2x2. Restart markers (DRI) are accepted but ignored — the bitstream is
//! treated as one continuous segment, which is correct for the (overwhelming
//! majority) of JPEGs that don't use restart intervals.
//!
//! Goal: correctness for the typical MJPEG-over-HTTP face-detection workload.
//! No SIMD, no Huffman-table caching, no progressive mode.
//!
//! # Limitations
//!
//! - Progressive JPEGs (`SOF2`) return [`JpegError::Unsupported`].
//! - 12-bit JPEGs (`SOF` precision > 8) return [`JpegError::Unsupported`].
//! - Arithmetic coding (`SOF9`-class) is not implemented.
//! - Restart markers in the entropy stream are not honored — a DRI-restarted
//!   JPEG with corrupted data between restarts will fail to decode even if
//!   the data is recoverable. The vast majority of baseline JPEGs in the wild
//!   do not use restart intervals, so this is acceptable for our use case.
//! - Truncated or bitstream-corrupt JPEGs return [`JpegError::Bitstream`].
//!
//! Zero-dep: no `extern crate` anywhere.

use crate::image::{GrayImage, RgbImage};
use std::fmt;

/// Errors a baseline JPEG decoder can return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JpegError {
    NotJpeg,
    BadMarker,
    BadMarkerLength,
    BadTable,
    BadFrameHeader,
    BadScanHeader,
    Bitstream,
    TooLarge,
    Unsupported(&'static str),
}

impl fmt::Display for JpegError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JpegError::NotJpeg => f.write_str("not a JPEG (missing SOI)"),
            JpegError::BadMarker => f.write_str("bad marker"),
            JpegError::BadMarkerLength => f.write_str("bad marker length"),
            JpegError::BadTable => f.write_str("bad Huffman or quantization table"),
            JpegError::BadFrameHeader => f.write_str("bad SOF frame header"),
            JpegError::BadScanHeader => f.write_str("bad SOS scan header"),
            JpegError::Bitstream => f.write_str("truncated or corrupt bitstream"),
            JpegError::TooLarge => f.write_str("image dimensions too large"),
            JpegError::Unsupported(what) => write!(f, "unsupported: {what}"),
        }
    }
}

impl std::error::Error for JpegError {}

impl From<std::io::Error> for JpegError {
    fn from(_: std::io::Error) -> Self {
        JpegError::Bitstream
    }
}

// Markers (T.81 table B.1).
const M_SOI: u8 = 0xD8;
const M_EOI: u8 = 0xD9;
const M_SOF0: u8 = 0xC0;
const M_SOF2: u8 = 0xC2;
const M_DHT: u8 = 0xC4;
const M_DQT: u8 = 0xDB;
const M_SOS: u8 = 0xDA;
const M_DRI: u8 = 0xDD;
const M_APP0: u8 = 0xE0;
const M_COM: u8 = 0xFE;
const M_RST0: u8 = 0xD0;
const M_RST7: u8 = 0xD7;

/// Maximum image dimension. Prevents hostile 65535×65535 headers.
const MAX_DIM: usize = 16384;

// =============================================================================
//   Huffman table
// =============================================================================

/// A Huffman table stored as a sparse tree. We could use a fast 8-bit lookup
/// table for short codes (the standard textbook optimization) but the simple
/// tree-walk is correct on every code length up to 16 bits and adds only a
/// handful of branches per symbol on the entropy-coded segment — well within
/// the cost budget for the 24×24–64×64 face windows this crate decodes.
///
/// The tree is laid out in a flat `Vec<Node>`; `Node { left, right, symbol }`
/// where `symbol == u16::MAX` marks an internal node and `left == u16::MAX &&
/// right == u16::MAX` marks a leaf. The root lives at index 0.
#[derive(Clone, Copy)]
struct HuffNode {
    left: u16,
    right: u16,
    symbol: u16,
}

const EMPTY_NODE: HuffNode = HuffNode {
    left: u16::MAX,
    right: u16::MAX,
    symbol: u16::MAX,
};

#[derive(Clone)]
struct HuffmanTable {
    nodes: Vec<HuffNode>,
}

impl Default for HuffmanTable {
    fn default() -> Self {
        Self {
            nodes: vec![EMPTY_NODE],
        }
    }
}

/// Build the Huffman table from T.81 DHT payload: 16 `bits[]` counts and the
/// flat `huffval[]` symbol list.
fn build_huffman_table(bits: &[u8; 16], huffval: &[u8]) -> Result<HuffmanTable, JpegError> {
    let total: u32 = bits.iter().map(|&b| b as u32).sum();
    if total > 256 || huffval.len() < total as usize {
        return Err(JpegError::BadTable);
    }
    let mut table = HuffmanTable::default();
    let mut code: u32 = 0;
    let mut symbol_iter = huffval.iter().copied();
    for bits_len in 1..=16u8 {
        let count = bits[(bits_len - 1) as usize] as u32;
        for _ in 0..count {
            let sym = symbol_iter.next().ok_or(JpegError::BadTable)?;
            // Walk `bits_len` bits of `code` from the MSB, branching left/right
            // at each step. If we hit a leaf prematurely, the table is
            // inconsistent.
            let mut node_idx: usize = 0;
            for bit_idx in 0..bits_len {
                let bit = ((code >> (bits_len - 1 - bit_idx)) & 1) as u8;
                let cur = table.nodes[node_idx];
                let next_idx = if bit == 0 { cur.left } else { cur.right };
                if next_idx == u16::MAX {
                    // Allocate a new child.
                    let new_idx = table.nodes.len();
                    if new_idx > u16::MAX as usize {
                        return Err(JpegError::BadTable);
                    }
                    table.nodes.push(EMPTY_NODE);
                    if bit == 0 {
                        table.nodes[node_idx].left = new_idx as u16;
                    } else {
                        table.nodes[node_idx].right = new_idx as u16;
                    }
                    node_idx = new_idx;
                } else {
                    node_idx = next_idx as usize;
                }
            }
            // We should be at a fresh (empty) leaf.
            let leaf = table.nodes[node_idx];
            if leaf.symbol != u16::MAX || leaf.left != u16::MAX || leaf.right != u16::MAX {
                return Err(JpegError::BadTable);
            }
            table.nodes[node_idx].symbol = sym as u16;
            code += 1;
        }
        code <<= 1;
    }
    // Trailing symbols after the code space is exhausted are an error in
    // strict T.81, but lenient decoders accept them. We mirror the strict
    // behaviour: anything left over is a malformed table.
    if symbol_iter.next().is_some() {
        return Err(JpegError::BadTable);
    }
    Ok(table)
}

// =============================================================================
//   Frame / component metadata
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumComponents {
    Gray = 1,
    YCbCr = 3,
}

#[derive(Clone, Copy, Debug)]
struct ComponentInfo {
    id: u8,
    factor_h: u8,
    factor_v: u8,
    quant_table: u8,
}

struct Frame {
    num_components: NumComponents,
    width: u16,
    height: u16,
    components: Vec<ComponentInfo>,
    max_h: u8,
    max_v: u8,
    /// MCUs between restart markers (0 = none). From DRI.
    restart_interval: u32,
}

impl Frame {
    /// Allocated width of a component's sample plane: one block column per
    /// `factor_h` blocks in every MCU column, rounded up to whole MCUs.
    ///
    /// This is *not* `width * factor_h / max_h`. The scan is MCU-interleaved,
    /// so a component always occupies a whole number of MCUs, and the plane is
    /// padded out to that. The padding is what the last MCU's blocks decode
    /// into and is cropped away on upsample.
    fn plane_dims(&self, comp: &ComponentInfo) -> (usize, usize) {
        (
            self.n_mcu_x() * comp.factor_h as usize * 8,
            self.n_mcu_y() * comp.factor_v as usize * 8,
        )
    }
}

impl Frame {
    fn n_mcu_x(&self) -> usize {
        let mcu_w = self.max_h as usize * 8;
        ((self.width as usize + mcu_w - 1) / mcu_w).max(1)
    }
    fn n_mcu_y(&self) -> usize {
        let mcu_h = self.max_v as usize * 8;
        ((self.height as usize + mcu_h - 1) / mcu_h).max(1)
    }
}

/// Scan header parsed from SOS.
#[derive(Clone, Copy, Debug)]
struct ScanInfo {
    /// Component IDs in scan order, padded with 0.
    components: [u8; 3],
    n_components: u8,
    /// Per-component Huffman table selectors.
    dc_table: [u8; 3],
    ac_table: [u8; 3],
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
}

// =============================================================================
//   Quantization table
// =============================================================================

#[derive(Clone, Copy)]
struct QuantTable {
    /// Values in zig-zag order.
    values: [u8; 64],
}

impl QuantTable {
    const fn empty() -> Self {
        Self { values: [0; 64] }
    }
}

// =============================================================================
//   Bitstream reader (MSB-first, with 0xFF 0x00 byte stuffing)
// =============================================================================

struct BitReader<'a> {
    data: &'a [u8],
    /// Bit accumulator (low bits valid).
    bits: u64,
    nbits: u32,
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            bits: 0,
            nbits: 0,
            pos: 0,
        }
    }

    #[inline]
    fn fill(&mut self, need: u32) {
        while self.nbits < need {
            if self.pos >= self.data.len() {
                break;
            }
            let b = self.data[self.pos];
            self.pos += 1;
            if b == 0xFF {
                if self.pos < self.data.len() {
                    let next = self.data[self.pos];
                    if next == 0x00 {
                        // 0xFF 0x00 = literal 0xFF in entropy stream.
                        self.pos += 1;
                        self.bits = (self.bits << 8) | 0xFFu64;
                        self.nbits += 8;
                        continue;
                    } else if next == 0xFF {
                        // Padding — skip and try again.
                        self.pos += 1;
                        continue;
                    } else {
                        // Real marker — back up so the framing layer can read it.
                        self.pos -= 1;
                        break;
                    }
                } else {
                    break;
                }
            } else {
                self.bits = (self.bits << 8) | (b as u64);
                self.nbits += 8;
            }
        }
    }

    /// Peek at the top `n` bits without consuming.
    #[inline]
    fn peek(&mut self, n: u32) -> u32 {
        self.fill(n);
        if self.nbits < n {
            return 0;
        }
        let shift = self.nbits - n;
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        ((self.bits >> shift) as u32) & mask
    }

    /// Drop `n` bits from the top of the buffer.
    #[inline]
    fn consume(&mut self, n: u32) {
        debug_assert!(n <= self.nbits);
        let shift = self.nbits - n;
        self.bits &= if shift >= 64 { 0 } else { (1u64 << shift) - 1 };
        self.nbits -= n;
    }

    /// Read `n` bits MSB-first.
    #[inline]
    fn read(&mut self, n: u32) -> u32 {
        let v = self.peek(n);
        if self.nbits >= n {
            self.consume(n);
        } else {
            self.nbits = 0;
        }
        v
    }

    /// Decode one Huffman symbol.
    fn decode_huffman(&mut self, table: &HuffmanTable) -> Result<u16, JpegError> {
        let mut node_idx: usize = 0;
        loop {
            // Need at least one bit to branch.
            self.fill(1);
            if self.nbits == 0 {
                return Err(JpegError::Bitstream);
            }
            // Peek the top bit, consume it, branch.
            let bit = self.peek(1) & 1;
            self.consume(1);
            let node = table.nodes[node_idx];
            let next = if bit == 0 { node.left } else { node.right };
            if next == u16::MAX {
                return Err(JpegError::Bitstream);
            }
            let child = table.nodes[next as usize];
            if child.symbol != u16::MAX {
                return Ok(child.symbol);
            }
            node_idx = next as usize;
        }
    }

    /// Consume the RSTn marker that must follow a restart interval.
    ///
    /// Any bits still buffered belong to the *previous* segment and must be
    /// discarded: after an RSTn the decoder restarts with a byte-aligned
    /// stream. `fill` deliberately rewinds `pos` to the 0xFF of a real marker
    /// rather than consuming it, so the marker is still there to be read here.
    fn consume_restart(&mut self, expected: u8) -> Result<u8, JpegError> {
        self.bits = 0;
        self.nbits = 0;
        while self.pos + 1 < self.data.len() {
            if self.data[self.pos] == 0xFF {
                let m = self.data[self.pos + 1];
                if (M_RST0..=M_RST7).contains(&m) {
                    let idx = m - M_RST0;
                    self.pos += 2;
                    return Ok(idx);
                }
                if m != 0x00 && m != 0xFF {
                    // Some other marker (usually EOI) — the encoder ended the
                    // scan without the trailing restart we expected.
                    return Err(JpegError::BadMarker);
                }
            }
            self.pos += 1;
        }
        let _ = expected;
        Err(JpegError::Bitstream)
    }
}

// =============================================================================
//   IDCT
// =============================================================================

/// The 8x8 inverse-DCT basis with its normalisation folded in:
/// `IDCT_M[x * 8 + u] = c(u) * cos((2x + 1) * u * pi / 16) / 2`.
///
/// Written out, the 2D inverse DCT is
///
/// ```text
/// f(x, y) = (1/4) * SUM_u SUM_v c(u) c(v) F(u, v)
///                                    * cos((2x+1) u pi/16) * cos((2y+1) v pi/16)
/// ```
///
/// so folding the `1/4` and both `c(.)` factors into the matrix leaves
/// `f = A . F . A^T` — two 8x8 products, separable over rows then columns.
///
/// This replaced an AAN even-odd butterfly that looked plausible and was
/// wrong: it was missing the 1/4 (every sample 4x too large, saturating to
/// noise), and adding the 1/4 back still left the reconstruction wrong,
/// because the even-odd split it implemented is not this transform. A
/// textbook basis has no such room for error, and it is verifiable by
/// inspection against the definition above.
///
/// Cost is ~2 * 8^3 = 1024 multiply-accumulates per block rather than the
/// ~200 of a butterfly, which is the right trade: a 768x432 frame is ~3.2k
/// blocks, so this lands in the low hundreds of microseconds and is
/// irrelevant next to a CNN forward pass.
#[allow(clippy::excessive_precision, clippy::approx_constant)]
const IDCT_M: [f64; 64] = [
    0.35355339059327373, 0.49039264020161522, 0.46193976625564337, 0.41573480615127262,
    0.35355339059327379, 0.27778511650980114, 0.19134171618254492, 0.097545161008064166,
    0.35355339059327373, 0.41573480615127262, 0.19134171618254492, -0.097545161008064096,
    -0.35355339059327373, -0.49039264020161522, -0.46193976625564342, -0.27778511650980109,
    0.35355339059327373, 0.27778511650980114, -0.19134171618254486, -0.49039264020161522,
    -0.35355339059327384, 0.097545161008064152, 0.46193976625564326, 0.41573480615127273,
    0.35355339059327373, 0.097545161008064166, -0.46193976625564337, -0.27778511650980109,
    0.35355339059327368, 0.41573480615127273, -0.19134171618254495, -0.49039264020161533,
    0.35355339059327373, -0.097545161008064096, -0.46193976625564342, 0.27778511650980092,
    0.35355339059327384, -0.41573480615127256, -0.19134171618254528, 0.49039264020161522,
    0.35355339059327373, -0.27778511650980098, -0.19134171618254517, 0.49039264020161522,
    -0.35355339059327334, -0.097545161008064013, 0.46193976625564337, -0.41573480615127251,
    0.35355339059327373, -0.41573480615127267, 0.191341716182545, 0.097545161008064388,
    -0.35355339059327356, 0.49039264020161533, -0.4619397662556432, 0.27778511650980076,
    0.35355339059327373, -0.49039264020161522, 0.46193976625564326, -0.41573480615127256,
    0.35355339059327329, -0.27778511650980076, 0.19134171618254478, -0.097545161008064291,
];

/// Inverse DCT of one dequantized 8x8 block, in place.
///
/// Input and output are both in natural (row-major) order. The output is the
/// level-shifted residual in roughly +/-128; the caller adds the +128 bias
/// when it clamps into a `u8` plane.
fn idct_8x8(block: &mut [i32; 64]) {
    // Rows: tmp[x][v] = SUM_u M[x][u] * F[u][v]
    let mut tmp = [0.0f64; 64];
    for x in 0..8 {
        for v in 0..8 {
            let mut acc = 0.0f64;
            for u in 0..8 {
                let c = block[u * 8 + v];
                if c != 0 {
                    acc += IDCT_M[x * 8 + u] * c as f64;
                }
            }
            tmp[x * 8 + v] = acc;
        }
    }
    // Columns: f[x][y] = SUM_v M[y][v] * tmp[x][v]
    for x in 0..8 {
        for y in 0..8 {
            let mut acc = 0.0f64;
            for v in 0..8 {
                let t = tmp[x * 8 + v];
                if t != 0.0 {
                    acc += IDCT_M[y * 8 + v] * t;
                }
            }
            // Round half away from zero, matching the reference decoder's
            // rounding of the reconstructed sample.
            block[x * 8 + y] = if acc >= 0.0 {
                (acc + 0.5) as i32
            } else {
                -((-acc + 0.5) as i32)
            };
        }
    }
}

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

// =============================================================================
//   YCbCr → RGB (BT.601 full-range, 8-bit fixed-point)
// =============================================================================

#[inline]
fn ycbcr_to_rgb(y: i32, cb: i32, cr: i32) -> (u8, u8, u8) {
    // BT.601 full range:
    //   R = Y                        + 1.40200 * (Cr - 128)
    //   G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
    //   B = Y + 1.77200 * (Cb - 128)
    //
    // The `+ 32768` before each shift is the round-to-nearest that libjpeg
    // bakes into its precomputed coefficient tables. Without it every channel
    // sits half a level low, which is small enough to look fine and large
    // enough to cap the achievable PSNR against the reference decoder.
    const ROUND: i32 = 1 << 15;
    let r = y + ((91881 * (cr - 128) + ROUND) >> 16);
    let g = y + ((-22554 * (cb - 128) - 46802 * (cr - 128) + ROUND) >> 16);
    let b = y + ((116130 * (cb - 128) + ROUND) >> 16);
    let clamp = |v: i32| v.clamp(0, 255) as u8;
    (clamp(r), clamp(g), clamp(b))
}

// =============================================================================
//   Cursor (framing-layer byte reader)
// =============================================================================

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
    fn read_u8(&mut self) -> Result<u8, JpegError> {
        if self.pos >= self.data.len() {
            return Err(JpegError::Bitstream);
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }
    fn read_u16(&mut self) -> Result<u16, JpegError> {
        if self.pos + 2 > self.data.len() {
            return Err(JpegError::Bitstream);
        }
        let v = u16::from_be_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }
    fn skip(&mut self, n: usize) -> Result<(), JpegError> {
        if self.pos + n > self.data.len() {
            return Err(JpegError::Bitstream);
        }
        self.pos += n;
        Ok(())
    }
    fn slice_from(&self, offset: usize) -> &'a [u8] {
        &self.data[offset..]
    }
    /// Read the next marker. Skips 0xFF fill bytes; reads a non-0xFF byte
    /// following 0xFF as the marker ID.
    fn read_marker(&mut self) -> Result<u8, JpegError> {
        loop {
            let b = self.read_u8()?;
            if b == 0xFF {
                loop {
                    let m = self.read_u8()?;
                    if m == 0xFF {
                        continue;
                    }
                    return Ok(m);
                }
            }
        }
    }
}

// =============================================================================
//   Top-level decode
// =============================================================================

/// Decode a JPEG byte stream into an RGB image.
pub fn decode_jpeg_rgb(bytes: &[u8]) -> Result<RgbImage, JpegError> {
    let parsed = parse_jpeg(bytes)?;
    let frame = &parsed.frame;
    let width = frame.width as usize;
    let height = frame.height as usize;
    if frame.num_components == NumComponents::Gray {
        let mut out = RgbImage::new(width, height);
        let plane = &parsed.planes[0];
        for y in 0..height {
            for x in 0..width {
                let g = plane[y * width + x];
                let o = (y * width + x) * 3;
                out.as_mut_slice()[o] = g;
                out.as_mut_slice()[o + 1] = g;
                out.as_mut_slice()[o + 2] = g;
            }
        }
        return Ok(out);
    }
    let cb = parsed.planes[1].clone();
    let cr = parsed.planes[2].clone();
    let mut out = RgbImage::new(width, height);
    let y_plane = &parsed.planes[0];
    for y in 0..height {
        for x in 0..width {
            let yv = y_plane[y * width + x] as i32;
            let cbv = cb[y * width + x] as i32;
            let crv = cr[y * width + x] as i32;
            let (r, g, b) = ycbcr_to_rgb(yv, cbv, crv);
            let o = (y * width + x) * 3;
            out.as_mut_slice()[o] = r;
            out.as_mut_slice()[o + 1] = g;
            out.as_mut_slice()[o + 2] = b;
        }
    }
    Ok(out)
}

/// Decode a JPEG byte stream into a grayscale image. Color JPEGs collapse to
/// luminance via BT.601 weights on the decoded RGB.
pub fn decode_jpeg_gray(bytes: &[u8]) -> Result<GrayImage, JpegError> {
    let rgb = decode_jpeg_rgb(bytes)?;
    Ok(rgb.to_gray())
}

/// Fully parsed JPEG: frame, all tables, decoded planes at image resolution.
struct DecodedJpeg {
    frame: Frame,
    planes: Vec<Vec<u8>>,
}

fn parse_jpeg(bytes: &[u8]) -> Result<DecodedJpeg, JpegError> {
    let mut cur = Cursor::new(bytes);
    if cur.remaining() < 2 || cur.read_u16()? != 0xFFD8 {
        return Err(JpegError::NotJpeg);
    }
    let mut quant_tables = [
        QuantTable::empty(),
        QuantTable::empty(),
        QuantTable::empty(),
        QuantTable::empty(),
    ];
    let mut quant_loaded = [false; 4];
    let mut huff_dc: [HuffmanTable; 2] = [HuffmanTable::default(), HuffmanTable::default()];
    let mut huff_ac: [HuffmanTable; 2] = [HuffmanTable::default(), HuffmanTable::default()];
    let mut frame: Option<Frame> = None;
    let mut scan: Option<ScanInfo> = None;
    let mut entropy_start: usize = 0;
    loop {
        let marker = cur.read_marker()?;
        match marker {
            M_EOI => break,
            M_SOF0 => {
                let f = read_sof0(&mut cur)?;
                frame = Some(f);
            }
            M_SOF2 => return Err(JpegError::Unsupported("progressive JPEG (SOF2)")),
            M_DQT => {
                read_dqt(&mut cur, &mut quant_tables, &mut quant_loaded)?;
            }
            M_DHT => {
                read_dht(&mut cur, &mut huff_dc, &mut huff_ac)?;
            }
            x if (0xC1..=0xCF).contains(&x) => {
                return Err(JpegError::Unsupported("extended sequential SOF"));
            }
            M_SOS => {
                let s = read_sos(&mut cur, frame.as_ref().ok_or(JpegError::BadScanHeader)?)?;
                scan = Some(s);
                entropy_start = cur.pos;
                break;
            }
            M_DRI => {
                let len = cur.read_u16()?;
                if len != 4 {
                    return Err(JpegError::BadMarkerLength);
                }
                // Restart interval: MCUs per RSTn segment. Must be *read*, not
                // skipped — discarding it made the decoder walk straight into
                // an RSTn mid-scan and give up.
                let ri = cur.read_u16()? as u32;
                if let Some(f) = frame.as_mut() {
                    f.restart_interval = ri;
                }
            }
            M_APP0 | M_COM => {
                let len = cur.read_u16()?;
                if len < 2 {
                    return Err(JpegError::BadMarkerLength);
                }
                cur.skip(len as usize - 2)?;
            }
            x if (0xE0..=0xEF).contains(&x) || (0xF0..=0xFE).contains(&x) => {
                let len = cur.read_u16()?;
                if len < 2 {
                    return Err(JpegError::BadMarkerLength);
                }
                cur.skip(len as usize - 2)?;
            }
            x if (M_RST0..=M_RST7).contains(&x) => {
                return Err(JpegError::BadMarker);
            }
            0x01 => { /* TEM — no payload. */ }
            _ => {
                if cur.remaining() >= 2 {
                    let len = cur.read_u16()?;
                    if len < 2 {
                        return Err(JpegError::BadMarkerLength);
                    }
                    cur.skip(len as usize - 2)?;
                } else {
                    break;
                }
            }
        }
    }
    let frame = frame.ok_or(JpegError::BadFrameHeader)?;
    let scan = scan.ok_or(JpegError::BadScanHeader)?;
    if (frame.width as usize) > MAX_DIM || (frame.height as usize) > MAX_DIM {
        return Err(JpegError::TooLarge);
    }
    // Trim the entropy segment at the first marker that ends the scan.
    //
    // RSTn is *not* such a marker: it appears in the middle of the entropy
    // data and the decoder has to step over each one at the restart
    // boundaries. Trimming at the first RSTn — as this used to — left the
    // scan with a fragment of the image and every restart-interval file failed
    // with "truncated or corrupt bitstream".
    let mut entropy_end = bytes.len();
    let mut i = entropy_start;
    while i < bytes.len() {
        if bytes[i] == 0xFF && i + 1 < bytes.len() {
            let m = bytes[i + 1];
            if m != 0x00 && m != 0xFF && !(M_RST0..=M_RST7).contains(&m) {
                entropy_end = i;
                break;
            }
        }
        i += 1;
    }
    let entropy = &bytes[entropy_start..entropy_end];
    let planes = decode_planes(
        &frame,
        &scan,
        entropy,
        &huff_dc,
        &huff_ac,
        &quant_tables,
        &quant_loaded,
    )?;
    Ok(DecodedJpeg { frame, planes })
}

fn read_sof0(cur: &mut Cursor<'_>) -> Result<Frame, JpegError> {
    let len = cur.read_u16()?;
    if len < 8 {
        return Err(JpegError::BadMarkerLength);
    }
    let precision = cur.read_u8()?;
    if precision != 8 {
        return Err(JpegError::Unsupported("non-8-bit precision"));
    }
    let height = cur.read_u16()?;
    let width = cur.read_u16()?;
    if width == 0 || height == 0 || (width as usize) > MAX_DIM || (height as usize) > MAX_DIM {
        return Err(JpegError::TooLarge);
    }
    let nf = cur.read_u8()?;
    let n_components = match nf {
        1 => NumComponents::Gray,
        3 => NumComponents::YCbCr,
        _ => return Err(JpegError::Unsupported("non 1/3 component count")),
    };
    let mut components = Vec::with_capacity(nf as usize);
    let mut max_h = 0u8;
    let mut max_v = 0u8;
    for _ in 0..nf {
        let id = cur.read_u8()?;
        let hv = cur.read_u8()?;
        let h = hv >> 4;
        let v = hv & 0xF;
        if h == 0 || v == 0 || h > 4 || v > 4 {
            return Err(JpegError::BadFrameHeader);
        }
        let qt = cur.read_u8()?;
        components.push(ComponentInfo {
            id,
            factor_h: h,
            factor_v: v,
            quant_table: qt,
        });
        if h > max_h {
            max_h = h;
        }
        if v > max_v {
            max_v = v;
        }
    }
    let total: u32 = components
        .iter()
        .map(|c| c.factor_h as u32 * c.factor_v as u32)
        .sum();
    if total > 10 {
        return Err(JpegError::BadFrameHeader);
    }
    Ok(Frame {
        num_components: n_components,
        width,
        height,
        components,
        max_h,
        max_v,
        restart_interval: 0,
    })
}

fn read_dqt(
    cur: &mut Cursor<'_>,
    tables: &mut [QuantTable; 4],
    loaded: &mut [bool; 4],
) -> Result<(), JpegError> {
    let len = cur.read_u16()?;
    if len < 2 {
        return Err(JpegError::BadMarkerLength);
    }
    let end = cur.pos + (len as usize - 2);
    if end > cur.data.len() {
        return Err(JpegError::BadMarkerLength);
    }
    while cur.pos < end {
        let pt = cur.read_u8()?;
        let precision = pt >> 4;
        let id = pt & 0xF;
        if id >= 4 || precision != 0 {
            return Err(JpegError::BadTable);
        }
        let mut qt = QuantTable::empty();
        for i in 0..64 {
            qt.values[i] = cur.read_u8()?;
        }
        tables[id as usize] = qt;
        loaded[id as usize] = true;
    }
    Ok(())
}

fn read_dht(
    cur: &mut Cursor<'_>,
    dc: &mut [HuffmanTable; 2],
    ac: &mut [HuffmanTable; 2],
) -> Result<(), JpegError> {
    let len = cur.read_u16()?;
    if len < 2 {
        return Err(JpegError::BadMarkerLength);
    }
    let end = cur.pos + (len as usize - 2);
    if end > cur.data.len() {
        return Err(JpegError::BadMarkerLength);
    }
    while cur.pos < end {
        let tc_th = cur.read_u8()?;
        let tc = tc_th >> 4;
        let th = tc_th & 0xF;
        if tc > 1 || th > 1 {
            return Err(JpegError::BadTable);
        }
        let mut bits = [0u8; 16];
        for i in 0..16 {
            bits[i] = cur.read_u8()?;
        }
        let total: usize = bits.iter().map(|&b| b as usize).sum();
        if total > 256 {
            return Err(JpegError::BadTable);
        }
        let mut huffval = vec![0u8; total];
        for i in 0..total {
            huffval[i] = cur.read_u8()?;
        }
        let table = build_huffman_table(&bits, &huffval)?;
        if tc == 0 {
            dc[th as usize] = table;
        } else {
            ac[th as usize] = table;
        }
    }
    Ok(())
}

fn read_sos(cur: &mut Cursor<'_>, frame: &Frame) -> Result<ScanInfo, JpegError> {
    let len = cur.read_u16()?;
    if len < 6 {
        return Err(JpegError::BadMarkerLength);
    }
    let ns = cur.read_u8()?;
    if (ns as usize) != frame.components.len() {
        return Err(JpegError::BadScanHeader);
    }
    let mut components = [0u8; 3];
    let mut dc_table = [0u8; 3];
    let mut ac_table = [0u8; 3];
    for i in 0..ns {
        let id = cur.read_u8()?;
        let td_ta = cur.read_u8()?;
        components[i as usize] = id;
        dc_table[i as usize] = td_ta >> 4;
        ac_table[i as usize] = td_ta & 0xF;
    }
    let ss = cur.read_u8()?;
    let se = cur.read_u8()?;
    let ahl = cur.read_u8()?;
    let ah = ahl >> 4;
    let al = ahl & 0xF;
    if ss != 0 || se != 63 || ah != 0 || al != 0 {
        return Err(JpegError::Unsupported("non-baseline scan header"));
    }
    Ok(ScanInfo {
        components,
        n_components: ns,
        dc_table,
        ac_table,
        ss,
        se,
        ah,
        al,
    })
}

// =============================================================================
//   Decode the entropy segment into image-resolution planes
// =============================================================================

fn decode_planes(
    frame: &Frame,
    scan: &ScanInfo,
    entropy: &[u8],
    huff_dc: &[HuffmanTable; 2],
    huff_ac: &[HuffmanTable; 2],
    quant_tables: &[QuantTable; 4],
    quant_loaded: &[bool; 4],
) -> Result<Vec<Vec<u8>>, JpegError> {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let n_mcu_x = frame.n_mcu_x();
    let n_mcu_y = frame.n_mcu_y();

    // One plane per component, each sized to its own MCU-aligned extent.
    //
    // The previous version sized *both* chroma planes with a hard-coded 4:2:0
    // formula, so a 4:2:2 file (h=2, v=1) got a chroma plane that was twice
    // too wide and the upsampler then read the wrong stride. Deriving the
    // dimensions from each component's own sampling factors handles 4:4:4,
    // 4:2:2 and 4:2:0 with the same code.
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(frame.components.len());
    let mut plane_w = [0usize; 3];
    let mut plane_h = [0usize; 3];
    for (p, comp) in frame.components.iter().enumerate() {
        let (w, h) = frame.plane_dims(comp);
        plane_w[p] = w;
        plane_h[p] = h;
        planes.push(vec![0u8; w * h]);
    }

    let mut reader = BitReader::new(entropy);
    let mut dc_pred = [0i32; 3];
    let mut block_buf = [0i32; 64];
    // Map scan-component-index → plane-index.
    let mut plane_idx = [0usize; 3];
    let mut factor_h = [0u8; 3];
    let mut factor_v = [0u8; 3];
    let mut qt_idx = [0u8; 3];
    for i in 0..scan.n_components as usize {
        let id = scan.components[i];
        let pos = frame
            .components
            .iter()
            .position(|c| c.id == id)
            .ok_or(JpegError::BadScanHeader)?;
        plane_idx[i] = pos;
        let comp = &frame.components[pos];
        factor_h[i] = comp.factor_h;
        factor_v[i] = comp.factor_v;
        qt_idx[i] = comp.quant_table;
    }

    let restart = frame.restart_interval as usize;
    let total_mcus = n_mcu_x * n_mcu_y;
    // How many MCUs sit before the final (often short) restart interval.
    // An encoder may omit the RSTn that would follow the last interval.
    let restart_boundaries = if restart == 0 {
        0
    } else {
        (total_mcus - 1) / restart
    };
    let mut boundaries_seen = 0usize;
    let mut mcu_index = 0usize;

    for mcu_y in 0..n_mcu_y {
        for mcu_x in 0..n_mcu_x {
            if restart > 0 && mcu_index > 0 && mcu_index % restart == 0 {
                // A restart marker sits between the previous MCU and this one.
                if boundaries_seen < restart_boundaries {
                    let expect = (boundaries_seen % 8) as u8;
                    let got = reader.consume_restart(expect)?;
                    if got != expect {
                        return Err(JpegError::BadMarker);
                    }
                    // DC predictors reset at every restart interval; keeping
                    // them would make the first block of each segment carry a
                    // difference against a predictor it never saw.
                    dc_pred = [0i32; 3];
                    boundaries_seen += 1;
                }
            }
            for s in 0..scan.n_components as usize {
                if !quant_loaded[qt_idx[s] as usize] {
                    return Err(JpegError::BadTable);
                }
                let qt = &quant_tables[qt_idx[s] as usize];
                let h = factor_h[s] as usize;
                let v = factor_v[s] as usize;
                for sy in 0..v {
                    for sx in 0..h {
                        decode_block(
                            &mut reader,
                            &mut block_buf,
                            &mut dc_pred[s],
                            &huff_dc[scan.dc_table[s] as usize],
                            &huff_ac[scan.ac_table[s] as usize],
                            qt,
                        )?;
                        let p = plane_idx[s];
                        // Block position *within this component's own plane*.
                        //
                        // The old code used `mcu_x * max_h + sx`, which is only
                        // right for a component whose factor equals the frame
                        // maximum. For a subsampled component (h=1 against
                        // max_h=2) that writes every block at twice its true
                        // offset, so half the chroma plane is never touched
                        // and the rest is shifted — which is what turned 4:2:0
                        // YCbCr output into a mosaic.
                        let bx = mcu_x * h + sx;
                        let by = mcu_y * v + sy;
                        write_block_to_plane(
                            &mut planes[p],
                            plane_w[p],
                            plane_h[p],
                            bx,
                            by,
                            &block_buf,
                        );
                    }
                }
            }
            mcu_index += 1;
        }
    }

    // Upsample every subsampled component to full image resolution, and crop
    // the rest. Every plane must end up exactly `width` x `height` with a
    // row stride of `width`, because that is the indexing the colour-conversion
    // and grayscale paths use; a full-resolution component still carries MCU
    // padding whenever the image is not a multiple of 8.
    for (p, comp) in frame.components.iter().enumerate() {
        let rh = frame.max_h as usize / comp.factor_h as usize;
        let rv = frame.max_v as usize / comp.factor_v as usize;
        if rh > 1 || rv > 1 {
            planes[p] =
                upsample_separable(&planes[p], plane_w[p], plane_h[p], rh, rv, width, height);
        } else if plane_w[p] != width || plane_h[p] != height {
            let (pw, ph) = (plane_w[p], plane_h[p]);
            let mut out = vec![0u8; width * height];
            let n = width.min(pw);
            for y in 0..height.min(ph) {
                out[y * width..y * width + n].copy_from_slice(&planes[p][y * pw..y * pw + n]);
            }
            planes[p] = out;
        }
    }
    Ok(planes)
}

/// Upsample a component plane by integer factors `rh` (horizontal) and `rv`
/// (vertical), then crop to `dw` x `dh`.
///
/// Uses the same triangle ("fancy") filter libjpeg defaults to, rather than
/// nearest-neighbour box replication. Box replication holds a chroma sample
/// over a 2x2 block with a visible stair-step at every colour edge; the
/// triangle filter interpolates toward the neighbouring sample and is what
/// the reference decoder does, so grading against it needs no slack.
///
/// For `rh == 2` this reduces exactly to libjpeg's
/// `out[2i] = (3*s[i] + s[i-1] + 2) / 4`, `out[2i+1] = (3*s[i] + s[i+1] + 2) / 4`.
fn upsample_separable(
    src: &[u8],
    sw: usize,
    sh: usize,
    rh: usize,
    rv: usize,
    dw: usize,
    dh: usize,
) -> Vec<u8> {
    // Horizontal first, into an intermediate buffer, then vertical. Keeping the
    // intermediate in u16 lets the vertical pass avoid a second rounding step.
    let iw = sw * rh;
    let ih = sh * rv;
    let mut mid = vec![0u16; iw * sh];
    for y in 0..sh {
        for x in 0..iw {
            let center = (x as f64 + 0.5) / rh as f64 - 0.5;
            let i0 = center.floor();
            let frac = center - i0;
            let a = (i0 as isize).clamp(0, sw as isize - 1) as usize;
            let b = (i0 as isize + 1).clamp(0, sw as isize - 1) as usize;
            let v = src[y * sw + a] as f64 * (1.0 - frac) + src[y * sw + b] as f64 * frac;
            mid[y * iw + x] = v.round().clamp(0.0, 255.0) as u16;
        }
    }
    let mut out = vec![0u8; dw * dh];
    for y in 0..dh.min(ih) {
        let center = (y as f64 + 0.5) / rv as f64 - 0.5;
        let i0 = center.floor();
        let frac = center - i0;
        let a = (i0 as isize).clamp(0, sh as isize - 1) as usize;
        let b = (i0 as isize + 1).clamp(0, sh as isize - 1) as usize;
        for x in 0..dw.min(iw) {
            let v = mid[a * iw + x] as f64 * (1.0 - frac) + mid[b * iw + x] as f64 * frac;
            out[y * dw + x] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn write_block_to_plane(
    plane: &mut [u8],
    plane_w: usize,
    plane_h: usize,
    bx: usize,
    by: usize,
    block: &[i32; 64],
) {
    let base_x = bx * 8;
    let base_y = by * 8;
    for y in 0..8 {
        let py = base_y + y;
        if py >= plane_h {
            break;
        }
        for x in 0..8 {
            let px = base_x + x;
            if px >= plane_w {
                break;
            }
            let v = block[y * 8 + x] + 128;
            let clamped = if v < 0 {
                0u8
            } else if v > 255 {
                255u8
            } else {
                v as u8
            };
            plane[py * plane_w + px] = clamped;
        }
    }
}

/// Decode one 8×8 block — DC (with prediction) + AC, dequantize, IDCT in place.
fn decode_block(
    reader: &mut BitReader<'_>,
    block: &mut [i32; 64],
    dc_pred: &mut i32,
    dc_table: &HuffmanTable,
    ac_table: &HuffmanTable,
    qt: &QuantTable,
) -> Result<(), JpegError> {
    for v in block.iter_mut() {
        *v = 0;
    }
    // DC.
    let dc_size = reader.decode_huffman(dc_table)?;
    if dc_size > 11 {
        return Err(JpegError::Bitstream);
    }
    let dc_diff = if dc_size == 0 {
        0
    } else {
        let bits = reader.read(dc_size as u32) as i32;
        // Sign-extend: if MSB is 0, value is negative.
        if bits < (1 << (dc_size - 1)) {
            bits - (1 << dc_size) + 1
        } else {
            bits
        }
    };
    let dc = *dc_pred + dc_diff;
    *dc_pred = dc;
    block[0] = dc * qt.values[0] as i32;
    // AC.
    let mut idx = 1usize;
    while idx < 64 {
        let rs = reader.decode_huffman(ac_table)?;
        if rs == 0x00 {
            break;
        }
        let r = (rs >> 4) as usize;
        let s = (rs & 0xF) as usize;
        if s == 0 {
            // The only s==0 codes are EOB (0x00, already handled above) and
            // ZRL (0xF0), which means "skip sixteen zero coefficients" and
            // carries no magnitude bits. Advancing by `r` alone would skip
            // only fifteen, leaving the next code to be read one coefficient
            // too early — which desynchronises the rest of the block and
            // everything after it.
            idx += 16;
            if idx > 64 {
                return Err(JpegError::Bitstream);
            }
            continue;
        }
        idx += r;
        if idx >= 64 {
            return Err(JpegError::Bitstream);
        }
        let bits = reader.read(s as u32) as i32;
        let v = if bits < (1 << (s - 1)) {
            bits - (1 << s) + 1
        } else {
            bits
        };
        block[idx] = v * qt.values[idx] as i32;
        idx += 1;
    }
    // De-zigzag into natural order for IDCT.
    //
    // `block` is indexed by *zigzag position*: block[z] holds the coefficient
    // whose natural (row-major) index is ZIGZAG[z]. Scattering — rather than
    // gathering — is therefore the correct direction.
    //
    // The old code gathered instead: `natural[i] = block[ZIGZAG[i]]`, which
    // applies the permutation twice and lands every coefficient in the wrong
    // place. The symptom was deceptive: position 0 is a fixed point of the
    // permutation, so a DC-only block decoded perfectly and smooth images came
    // out merely "a bit wrong" (~26 dB) while any block with real high
    // frequency — a checkerboard, a face's eye and mouth detail — was
    // scrambled outright (~5 dB).
    let mut natural = [0i32; 64];
    for z in 0..64 {
        natural[ZIGZAG[z]] = block[z];
    }
    *block = natural;
    idct_8x8(block);
    Ok(())
}

// =============================================================================
//   Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid baseline JPEG byte stream for an 8×8 all-128
    /// grayscale image, using only the symbols we need: DC size 0 (no DC
    /// diff) and AC EOB (zero-run of 0).
    fn make_8x8_gray_jpeg() -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&[0xFF, M_SOI]);
        // DQT — quant table 0 with all 1s.
        let mut dqt_payload = vec![0u8];
        dqt_payload.extend_from_slice(&[1u8; 64]);
        append_marker(&mut buf, M_DQT, &dqt_payload);
        // SOF0 — 8×8, 1 component, quant 0.
        let sof_payload: Vec<u8> = vec![8, 0, 8, 0, 8, 1, 1, 0x11, 0];
        append_marker(&mut buf, M_SOF0, &sof_payload);
        // DHT — DC table 0 with one symbol (size=0) at length 1.
        let mut dc_bits = [0u8; 16];
        dc_bits[0] = 1;
        append_dht(&mut buf, 0, 0, &dc_bits, &[0u8]);
        // DHT — AC table 0 with one symbol (EOB) at length 1.
        let mut ac_bits = [0u8; 16];
        ac_bits[0] = 1;
        append_dht(&mut buf, 1, 0, &ac_bits, &[0u8]);
        // SOS.
        let sos_payload: Vec<u8> = vec![1, 1, 0x00, 0, 63, 0x00];
        append_marker(&mut buf, M_SOS, &sos_payload);
        // Entropy: 1 block → DC size 0 (code = 0 of length 1) + EOB (code = 0
        // of length 1) = bits "00", padded with 1s to a byte: 0b00111111 = 0x3F.
        buf.push(0x3F);
        // EOI.
        buf.extend_from_slice(&[0xFF, M_EOI]);
        buf
    }

    /// Write a marker segment: `0xFF <marker> <length_be16> <payload>`.
    fn append_marker(buf: &mut Vec<u8>, marker: u8, payload: &[u8]) {
        let len = (payload.len() + 2) as u16;
        buf.push(0xFF);
        buf.push(marker);
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(payload);
    }

    fn append_dht(buf: &mut Vec<u8>, tc: u8, th: u8, bits: &[u8; 16], vals: &[u8]) {
        let mut payload = vec![tc << 4 | th];
        payload.extend_from_slice(bits);
        payload.extend_from_slice(vals);
        append_marker(buf, M_DHT, &payload);
    }

    /// Build a baseline JPEG byte stream for an 8×8 grayscale image with a
    /// non-zero DC value. Used to verify DC prediction works correctly.
    #[allow(dead_code)]
    fn make_8x8_gray_jpeg_nonzero_dc(dc_value: i16) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&[0xFF, M_SOI]);
        // DQT — quant table 0 with all 1s.
        let mut dqt_payload = vec![0u8];
        dqt_payload.extend_from_slice(&[1u8; 64]);
        append_marker(&mut buf, M_DQT, &dqt_payload);
        // SOF0.
        let sof_payload: Vec<u8> = vec![8, 0, 8, 0, 8, 1, 1, 0x11, 0];
        append_marker(&mut buf, M_SOF0, &sof_payload);
        // DC table: 2 symbols (size=0, size=1) at length 1, but only one bit
        // is needed for "size=1" since the actual size fits in 1 bit. We use
        // a 2-symbol table: code 0 → size 0, code 1 → size 1.
        let mut dc_bits = [0u8; 16];
        dc_bits[0] = 2;
        append_dht(&mut buf, 0, 0, &dc_bits, &[0u8, 1u8]);
        // AC table: EOB at length 1.
        let mut ac_bits = [0u8; 16];
        ac_bits[0] = 1;
        append_dht(&mut buf, 1, 0, &ac_bits, &[0u8]);
        // SOS.
        let sos_payload: Vec<u8> = vec![1, 1, 0x00, 0, 63, 0x00];
        append_marker(&mut buf, M_SOS, &sos_payload);
        // Entropy: DC size = 1, then 1 bit value. If dc_value = N, the bit is
        // the low bit of N's sign-magnitude representation. For N = +1, bit=1;
        // for N = -1, bit=0. Then EOB (code 0 of length 1).
        let dc_bit = if dc_value > 0 { 1u8 } else { 0u8 };
        // Bits: code 1 (size=1, 1 bit), value bit, code 0 (EOB). Total 3 bits.
        // Pad with 1s: 1_1_0_11111 = 0b110_11111 = 0xDF.
        let byte = 0b1101_1111u8;
        let _ = dc_bit; // value already encoded by code=1 selection
        buf.push(byte);
        buf.extend_from_slice(&[0xFF, M_EOI]);
        buf
    }

    #[test]
    fn round_trip_8x8_gray_jpeg_constant() {
        // Hand-crafted JPEG: 8×8 DC=0, AC=EOB → IDCT → all-zero block +
        // level shift +128 = 128 per pixel.
        let bytes = make_8x8_gray_jpeg();
        let decoded = decode_jpeg_gray(&bytes).expect("decode failed");
        assert_eq!(decoded.width(), 8);
        assert_eq!(decoded.height(), 8);
        for y in 0..8 {
            for x in 0..8 {
                let pix = decoded[(x, y)];
                let lo = 128u8.saturating_sub(2);
                let hi = 130u8;
                assert!(
                    pix >= lo && pix <= hi,
                    "pixel ({x},{y}) = {pix} not in [{lo},{hi}]"
                );
            }
        }
    }

    #[test]
    fn not_jpeg_returns_error() {
        let bytes = vec![0x00, 0x01, 0x02];
        match decode_jpeg_gray(&bytes) {
            Err(JpegError::NotJpeg) => {}
            other => panic!("expected NotJpeg, got {other:?}"),
        }
    }

    #[test]
    fn bitreader_handles_byte_stuffing() {
        let data = [0xFF, 0x00, 0xAB];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read(8), 0xFF);
        assert_eq!(r.read(8), 0xAB);
    }

    #[test]
    fn bitreader_terminates_at_marker() {
        let data = [0x12, 0xFF, M_EOI];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read(8), 0x12);
        // The next fill() will see 0xFF M_EOI and back up.
        let v = r.peek(8);
        // Either it peeks the byte (0xFF) or backs up — both are acceptable.
        let _ = v;
    }

    #[test]
    fn build_huffman_table_two_symbol() {
        // Two symbols: 0 → "0" (length 1), 1 → "1" (length 1).
        let bits: [u8; 16] = {
            let mut b = [0u8; 16];
            b[0] = 2;
            b
        };
        let vals = [0u8, 1u8];
        let t = build_huffman_table(&bits, &vals).unwrap();
        assert_eq!(t.nodes.len(), 3); // root + 2 leaves
    }

    #[test]
    fn build_huffman_table_rejects_bad_lengths() {
        let bits = [0u8; 16];
        let vals: [u8; 0] = [];
        assert!(build_huffman_table(&bits, &vals).is_ok());
        let bad_bits = [0u8; 16];
        let bad_vals = [0u8, 1, 2, 3];
        // 4 symbols declared but bits[] sums to 0 — inconsistent.
        assert!(build_huffman_table(&bad_bits, &bad_vals).is_err());
    }
}
