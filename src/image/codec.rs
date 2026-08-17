//! PPM/PGM codec — trivial binary format, useful for tests and intermediate frames.
//!
//! P5 (binary PGM) for grayscale, P6 (binary PPM) for RGB.
//!
//! We read the entire stream into memory and parse, which avoids fragile
//! one-byte-at-a-time reading.

use crate::image::{GrayImage, RgbImage};
use std::io::{Read, Write};

/// Read all remaining bytes from `r`.
fn slurp(r: &mut dyn Read) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Parse a whitespace-or-comment-separated decimal integer starting at `pos`.
fn parse_uint(buf: &[u8], mut pos: usize) -> Option<(usize, usize)> {
    // Skip whitespace and comments.
    while pos < buf.len() {
        let b = buf[pos];
        if b.is_ascii_whitespace() { pos += 1; continue; }
        if b == b'#' {
            while pos < buf.len() && buf[pos] != b'\n' { pos += 1; }
            continue;
        }
        break;
    }
    let start = pos;
    while pos < buf.len() && buf[pos].is_ascii_digit() { pos += 1; }
    if pos == start { return None; }
    let s = std::str::from_utf8(&buf[start..pos]).ok()?;
    let n: usize = s.parse().ok()?;
    Some((n, pos))
}

pub fn read_pgm(r: &mut dyn Read) -> std::io::Result<GrayImage> {
    let buf = slurp(r)?;
    if buf.len() < 2 || &buf[0..2] != b"P5" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "expected P5"));
    }
    let mut pos = 2;
    let (w, p) = parse_uint(&buf, pos).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad width"))?; pos = p;
    let (h, p) = parse_uint(&buf, pos).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad height"))?; pos = p;
    let (maxv, p) = parse_uint(&buf, pos).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad maxval"))?; pos = p;
    if maxv != 255 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "maxval must be 255"));
    }
    // Skip exactly one whitespace byte separating header from pixel data.
    if pos >= buf.len() || !buf[pos].is_ascii_whitespace() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "expected whitespace before pixel data"));
    }
    pos += 1;
    let needed = w * h;
    if buf.len() < pos + needed {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "pixel data truncated"));
    }
    let data = buf[pos..pos + needed].to_vec();
    Ok(GrayImage::from_vec(data, w, h))
}

pub fn read_ppm(r: &mut dyn Read) -> std::io::Result<RgbImage> {
    let buf = slurp(r)?;
    if buf.len() < 2 || &buf[0..2] != b"P6" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "expected P6"));
    }
    let mut pos = 2;
    let (w, p) = parse_uint(&buf, pos).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad width"))?; pos = p;
    let (h, p) = parse_uint(&buf, pos).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad height"))?; pos = p;
    let (maxv, p) = parse_uint(&buf, pos).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad maxval"))?; pos = p;
    if maxv != 255 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "maxval must be 255"));
    }
    if pos >= buf.len() || !buf[pos].is_ascii_whitespace() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "expected whitespace before pixel data"));
    }
    pos += 1;
    let needed = w * h * 3;
    if buf.len() < pos + needed {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "pixel data truncated"));
    }
    let data = buf[pos..pos + needed].to_vec();
    Ok(RgbImage { data, width: w, height: h })
}

pub fn write_pgm(w: &mut dyn Write, img: &GrayImage) -> std::io::Result<()> {
    write!(w, "P5\n{} {}\n255\n", img.width(), img.height())?;
    w.write_all(img.as_slice())
}

pub fn write_ppm(w: &mut dyn Write, img: &RgbImage) -> std::io::Result<()> {
    write!(w, "P6\n{} {}\n255\n", img.width(), img.height())?;
    w.write_all(img.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn pgm_roundtrip() {
        let mut img = GrayImage::new(4, 3);
        for y in 0..3 { for x in 0..4 { img[(x, y)] = ((x + y * 4) * 17) as u8; } }
        let mut buf = Vec::new();
        write_pgm(&mut buf, &img).unwrap();
        let back = read_pgm(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(back.width(), 4);
        assert_eq!(back.height(), 3);
        for y in 0..3 { for x in 0..4 { assert_eq!(img[(x, y)], back[(x, y)]); } }
    }

    #[test]
    fn ppm_roundtrip() {
        let mut img = RgbImage::new(2, 2);
        for y in 0..2 { for x in 0..2 {
            let r = img.row_mut(y);
            r[x*3] = (x*30 + y) as u8;
            r[x*3+1] = (y*40 + x) as u8;
            r[x*3+2] = 128;
        }}
        let mut buf = Vec::new();
        write_ppm(&mut buf, &img).unwrap();
        let back = read_ppm(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(back.width(), 2);
        assert_eq!(back.height(), 2);
        for y in 0..2 { for x in 0..2 {
            assert_eq!(img.row(y)[x*3], back.row(y)[x*3]);
            assert_eq!(img.row(y)[x*3+1], back.row(y)[x*3+1]);
            assert_eq!(img.row(y)[x*3+2], back.row(y)[x*3+2]);
        }}
    }
}
