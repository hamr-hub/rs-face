//! Minimal zero-dependency ZIP writer (STORE method, no compression).
//!
//! Used only by the `download.zip` export endpoint. Hand-rolled rather than
//! pulling in a `zip` crate: the packaged payload (PNG frames, face crops,
//! original media) is already compressed, so storing it uncompressed loses
//! nothing, and the platform gains no new dependency.
//!
//! Implements the smallest useful subset of APPNOTE 4.3.16:
//!
//! - one local file header + raw data per entry,
//! - central directory,
//! - end-of-central-directory record.
//!
//! All multi-byte fields are little-endian. Names are UTF-8 (general-purpose
//! bit 11 is set in both headers).

use std::io::Result;

const LOCAL_FILE_HEADER_SIG: u32 = 0x04034b50;
const CENTRAL_FILE_HEADER_SIG: u32 = 0x02014b50;
const END_OF_CENTRAL_DIR_SIG: u32 = 0x06054b50;

/// Version 2.0 — sufficient for STORE plus the UTF-8-name flag.
const VERSION_NEEDED: u16 = 20;
/// General-purpose bit 11: file name / comment are UTF-8.
const UTF8_FLAG: u16 = 0x0800;
/// DOS date 1980-01-01 and fixed time 00:00:00 — valid, deterministic stamps.
const DOS_DATE_1980_01_01: u16 = 0x0021;
const DOS_TIME_MIDNIGHT: u16 = 0;

#[derive(Clone)]
struct EntryMeta {
    name: Vec<u8>,
    crc: u32,
    size: u32,
    local_header_offset: u32,
}

/// Append-only in-memory ZIP builder. Call [`add_file`](Self::add_file) for each
/// entry, then [`finish`](Self::finish) to obtain the archive bytes.
#[derive(Default)]
pub(crate) struct ZipWriter {
    buf: Vec<u8>,
    entries: Vec<EntryMeta>,
}

impl ZipWriter {
    /// Create an empty writer.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Append one stored (uncompressed) file under `name`.
    ///
    /// Returns an error if `name` is empty, `data` exceeds 4 GiB, or the archive
    /// would exceed either the 4 GiB total or the 65 535-entry limit.
    pub(crate) fn add_file(&mut self, name: &str, data: &[u8]) -> Result<()> {
        if name.is_empty() {
            return Err(other("zip entry name is empty"));
        }
        if data.len() > u32::MAX as usize {
            return Err(other("zip entry exceeds 4 GiB"));
        }
        if self.entries.len() >= u16::MAX as usize {
            return Err(other("zip has too many entries"));
        }
        if self.buf.len().checked_add(data.len()).is_none()
            || self.buf.len() + data.len() > u32::MAX as usize
        {
            return Err(other("zip archive exceeds 4 GiB"));
        }

        let name_bytes = name.as_bytes();
        let crc = crc32(data);
        let size = data.len() as u32;
        let local_header_offset = self.buf.len() as u32;

        // Local file header.
        self.buf
            .extend_from_slice(&LOCAL_FILE_HEADER_SIG.to_le_bytes());
        self.buf.extend_from_slice(&VERSION_NEEDED.to_le_bytes());
        self.buf.extend_from_slice(&UTF8_FLAG.to_le_bytes());
        self.buf.extend_from_slice(&0u16.to_le_bytes()); // compression method 0 = store
        self.buf.extend_from_slice(&DOS_TIME_MIDNIGHT.to_le_bytes());
        self.buf
            .extend_from_slice(&DOS_DATE_1980_01_01.to_le_bytes());
        self.buf.extend_from_slice(&crc.to_le_bytes());
        self.buf.extend_from_slice(&size.to_le_bytes()); // compressed size
        self.buf.extend_from_slice(&size.to_le_bytes()); // uncompressed size
        let name_len = name_bytes.len() as u16;
        self.buf.extend_from_slice(&name_len.to_le_bytes());
        self.buf.extend_from_slice(&0u16.to_le_bytes()); // extra field length
        self.buf.extend_from_slice(name_bytes);
        self.buf.extend_from_slice(data);

        self.entries.push(EntryMeta {
            name: name_bytes.to_vec(),
            crc,
            size,
            local_header_offset,
        });
        Ok(())
    }

    /// Consume the writer and return the complete ZIP archive bytes.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        let central_dir_offset = self.buf.len() as u32;

        for e in &self.entries {
            // Central directory file header.
            self.buf
                .extend_from_slice(&CENTRAL_FILE_HEADER_SIG.to_le_bytes());
            self.buf.extend_from_slice(&VERSION_NEEDED.to_le_bytes()); // version made by
            self.buf.extend_from_slice(&VERSION_NEEDED.to_le_bytes()); // version needed
            self.buf.extend_from_slice(&UTF8_FLAG.to_le_bytes());
            self.buf.extend_from_slice(&0u16.to_le_bytes()); // method = store
            self.buf.extend_from_slice(&DOS_TIME_MIDNIGHT.to_le_bytes());
            self.buf
                .extend_from_slice(&DOS_DATE_1980_01_01.to_le_bytes());
            self.buf.extend_from_slice(&e.crc.to_le_bytes());
            self.buf.extend_from_slice(&e.size.to_le_bytes());
            self.buf.extend_from_slice(&e.size.to_le_bytes());
            self.buf
                .extend_from_slice(&(e.name.len() as u16).to_le_bytes());
            self.buf.extend_from_slice(&0u16.to_le_bytes()); // extra
            self.buf.extend_from_slice(&0u16.to_le_bytes()); // comment
            self.buf.extend_from_slice(&0u16.to_le_bytes()); // disk number start
            self.buf.extend_from_slice(&0u16.to_le_bytes()); // internal file attrs
            self.buf.extend_from_slice(&0u32.to_le_bytes()); // external file attrs
            self.buf
                .extend_from_slice(&e.local_header_offset.to_le_bytes());
            self.buf.extend_from_slice(&e.name);
        }

        let central_dir_size = self.buf.len() as u32 - central_dir_offset;
        let entry_count = self.entries.len() as u16;

        // End of central directory record.
        self.buf
            .extend_from_slice(&END_OF_CENTRAL_DIR_SIG.to_le_bytes());
        self.buf.extend_from_slice(&0u16.to_le_bytes()); // disk number
        self.buf.extend_from_slice(&0u16.to_le_bytes()); // disk with central dir
        self.buf.extend_from_slice(&entry_count.to_le_bytes()); // entries on this disk
        self.buf.extend_from_slice(&entry_count.to_le_bytes()); // total entries
        self.buf.extend_from_slice(&central_dir_size.to_le_bytes());
        self.buf
            .extend_from_slice(&central_dir_offset.to_le_bytes());
        self.buf.extend_from_slice(&0u16.to_le_bytes()); // comment length
        self.buf
    }
}

/// Standard CRC-32 (IEEE 802.3), reflected, as required by the ZIP format.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let idx = (crc ^ u32::from(byte)) & 0xFF;
        crc = (crc >> 8) ^ CRC_TABLE[idx as usize];
    }
    crc ^ 0xFFFF_FFFF
}

/// Build the 256-entry CRC lookup table once, lazily, via a const computation is
/// verbose; a static initialized by a small helper keeps this dependency-free.
static CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut n = 0usize;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            if c & 1 != 0 {
                c = (c >> 1) ^ 0xEDB8_8320;
            } else {
                c >>= 1;
            }
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
};

fn other(msg: &'static str) -> std::io::Error {
    std::io::Error::other(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_known_vectors() {
        // Standard CRC-32 check value (IEEE) and empty-input identity.
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn archive_has_local_central_and_eocd() {
        let mut zip = ZipWriter::new();
        zip.add_file("a.txt", b"hello").unwrap();
        zip.add_file("dir/b.bin", &[0u8, 1, 2, 3]).unwrap();
        let out = zip.finish();

        // Two local file headers and two central headers, in that order.
        let sig_at = |w: &&[u8]| u32::from_le_bytes(<[u8; 4]>::try_from(*w).unwrap());
        let locals = out
            .windows(4)
            .filter(|w| sig_at(w) == LOCAL_FILE_HEADER_SIG)
            .count();
        let centrals = out
            .windows(4)
            .filter(|w| sig_at(w) == CENTRAL_FILE_HEADER_SIG)
            .count();
        assert_eq!(locals, 2);
        assert_eq!(centrals, 2);

        // EOCD is the final 22 bytes and reports 2 entries.
        assert_eq!(
            u32::from_le_bytes(out[out.len() - 22..out.len() - 18].try_into().unwrap()),
            END_OF_CENTRAL_DIR_SIG
        );
        assert_eq!(
            u16::from_le_bytes(out[out.len() - 12..out.len() - 10].try_into().unwrap()),
            2
        );
    }

    #[test]
    fn empty_name_and_empty_archive_are_sane() {
        let mut zip = ZipWriter::new();
        assert!(zip.add_file("", b"x").is_err());
        let out = zip.finish();
        // An empty archive still carries an EOCD record.
        assert!(out.len() >= 22);
        assert_eq!(
            u32::from_le_bytes(out[out.len() - 22..out.len() - 18].try_into().unwrap()),
            END_OF_CENTRAL_DIR_SIG
        );
    }
}
