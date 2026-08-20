//! Minimal PNG encoder/decoder (no compression — uses stored DEFLATE blocks).
//!
//! Supports 8-bit grayscale and 8-bit RGB (color types 0 and 2).
//! Decoding handles all five filter types: None, Sub, Up, Average, Paeth.
//!
//! Produced files are larger than `zlib`-compressed output but valid.

use crate::image::{GrayImage, RgbImage};
use std::io::{Read, Write};

const PNG_SIG: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

fn crc32_update(mut crc: u32, buf: &[u8]) -> u32 {
    for &b in buf {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                0xEDB88320 ^ (crc >> 1)
            } else {
                crc >> 1
            };
        }
    }
    crc
}

fn make_chunk(typ: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + data.len() + 4);
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(typ);
    out.extend_from_slice(data);
    let crc = crc32_update(0xFFFFFFFFu32, typ);
    let crc = crc32_update(crc, data);
    out.extend_from_slice(&(crc ^ 0xFFFFFFFFu32).to_be_bytes());
    out
}

// ----- DEFLATE stored blocks (zlib wrapper) -----
fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &d in data {
        a = (a + d as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn write_deflate_stored(w: &mut dyn Write, data: &[u8]) -> std::io::Result<()> {
    // zlib header: CMF=0x78, FLG=0x01 (no dict, level 0)
    w.write_all(&[0x78, 0x01])?;
    let mut pos = 0;
    let max_chunk = 0xFFFF;
    while pos < data.len() {
        let end = (pos + max_chunk).min(data.len());
        let len = (end - pos) as u16;
        let final_block = end == data.len();
        let header = if final_block { 0x01u8 } else { 0x00u8 };
        w.write_all(&[header])?;
        w.write_all(&len.to_le_bytes())?;
        w.write_all(&(!len).to_le_bytes())?;
        w.write_all(&data[pos..end])?;
        pos = end;
    }
    w.write_all(&adler32(data).to_be_bytes())?;
    Ok(())
}

// ----- Encoder -----
pub fn write_png_gray(w: &mut dyn Write, img: &GrayImage) -> std::io::Result<()> {
    w.write_all(&PNG_SIG)?;
    // IHDR
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(img.width() as u32).to_be_bytes());
    ihdr.extend_from_slice(&(img.height() as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 0, 0, 0, 0]); // bit depth, color type (0=gray), compression, filter, interlace
    w.write_all(&make_chunk(b"IHDR", &ihdr))?;

    // IDAT: filter byte 0 per scanline + raw pixel data.
    let stride = img.width();
    let mut raw = Vec::with_capacity((stride + 1) * img.height());
    for y in 0..img.height() {
        raw.push(0u8); // filter: None
        raw.extend_from_slice(img.row(y));
    }
    let mut compressed = Vec::new();
    write_deflate_stored(&mut compressed, &raw)?;
    w.write_all(&make_chunk(b"IDAT", &compressed))?;
    w.write_all(&make_chunk(b"IEND", &[]))?;
    Ok(())
}

pub fn write_png_rgb(w: &mut dyn Write, img: &RgbImage) -> std::io::Result<()> {
    w.write_all(&PNG_SIG)?;
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(img.width() as u32).to_be_bytes());
    ihdr.extend_from_slice(&(img.height() as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // color type 2 = RGB
    w.write_all(&make_chunk(b"IHDR", &ihdr))?;

    let stride = img.width() * 3;
    let mut raw = Vec::with_capacity((stride + 1) * img.height());
    for y in 0..img.height() {
        raw.push(0u8);
        raw.extend_from_slice(img.row(y));
    }
    let mut compressed = Vec::new();
    write_deflate_stored(&mut compressed, &raw)?;
    w.write_all(&make_chunk(b"IDAT", &compressed))?;
    w.write_all(&make_chunk(b"IEND", &[]))?;
    Ok(())
}

// ----- Decoder -----
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}
impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn read_byte(&mut self) -> u8 {
        let b = self.data[self.pos];
        self.pos += 1;
        b
    }
    fn read_u16_le(&mut self) -> u16 {
        let l = self.read_byte() as u16;
        let h = self.read_byte() as u16;
        l | (h << 8)
    }
    fn skip_to_byte_boundary(&mut self) {
        if self.pos % 8 != 0 {
            self.pos = (self.pos + 7) / 8 * 8;
        }
    }
}

fn inflate_stored(r: &mut BitReader, out: &mut Vec<u8>) -> std::io::Result<()> {
    loop {
        r.skip_to_byte_boundary();
        let header = r.read_byte();
        let final_block = (header & 1) != 0;
        let btype = (header >> 1) & 3;
        if btype != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "only stored blocks supported",
            ));
        }
        let len = r.read_u16_le();
        let nlen = r.read_u16_le();
        if len as u16 != !nlen {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad LEN/NLEN",
            ));
        }
        let start = out.len();
        out.resize(start + len as usize, 0);
        out[start..start + len as usize].copy_from_slice(&r.data[r.pos..r.pos + len as usize]);
        r.pos += len as usize;
        if final_block {
            return Ok(());
        }
    }
}

fn unfilter_paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i32 + b as i32 - c as i32;
    let pa = (p - a as i32).abs();
    let pb = (p - b as i32).abs();
    let pc = (p - c as i32).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

pub fn read_png(r: &mut dyn Read) -> std::io::Result<(usize, usize, u8, Vec<u8>)> {
    // Returns (width, height, color_type, raw pixel bytes).
    let mut sig = [0u8; 8];
    r.read_exact(&mut sig)?;
    if sig != PNG_SIG {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad PNG signature",
        ));
    }
    let mut ihdr: Option<(usize, usize, u8, u8)> = None;
    let mut compressed = Vec::new();
    loop {
        let mut len_buf = [0u8; 4];
        r.read_exact(&mut len_buf)?;
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut typ = [0u8; 4];
        r.read_exact(&mut typ)?;
        let mut data = vec![0u8; len];
        r.read_exact(&mut data)?;
        let mut crc_buf = [0u8; 4];
        r.read_exact(&mut crc_buf)?;
        match &typ {
            b"IHDR" => {
                let w = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
                let h = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
                let depth = data[8];
                let ctype = data[9];
                if depth != 8 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "only 8-bit supported",
                    ));
                }
                ihdr = Some((w, h, ctype, depth));
            }
            b"IDAT" => compressed.extend_from_slice(&data),
            b"IEND" => break,
            _ => {} // skip ancillary chunks
        }
    }
    let (w, h, ctype, _depth) =
        ihdr.ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing IHDR"))?;

    // Decompress zlib: skip 2-byte header, then DEFLATE.
    if compressed.len() < 6 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "compressed data too short",
        ));
    }
    let zlib_body = &compressed[2..compressed.len() - 4];
    let mut raw = Vec::new();
    let mut br = BitReader::new(zlib_body);
    inflate_stored(&mut br, &mut raw)?;

    let bpp = match ctype {
        0 => 1,
        2 => 3,
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unsupported color type",
            ))
        }
    };
    let stride = w * bpp;
    let mut pixels = vec![0u8; stride * h];
    // Scratch buffer for "previous row" when y == 0 (filled with zeros).
    let mut prev_buf: Vec<u8> = vec![0u8; stride];
    for y in 0..h {
        let filter = raw[y * (stride + 1)];
        let cur_start = y * stride;
        let line = &mut pixels[cur_start..cur_start + stride];
        let scan = &raw[y * (stride + 1) + 1..(y + 1) * (stride + 1)];
        match filter {
            0 => line.copy_from_slice(scan),
            1 => {
                for x in 0..stride {
                    let left = if x >= bpp { line[x - bpp] } else { 0 };
                    line[x] = scan[x].wrapping_add(left);
                }
            }
            2 => {
                for x in 0..stride {
                    let up = prev_buf[x];
                    line[x] = scan[x].wrapping_add(up);
                }
            }
            3 => {
                for x in 0..stride {
                    let left = if x >= bpp { line[x - bpp] as u32 } else { 0 };
                    let up = prev_buf[x] as u32;
                    let pred = ((left + up) / 2) as u8;
                    line[x] = scan[x].wrapping_add(pred);
                }
            }
            4 => {
                for x in 0..stride {
                    let left = if x >= bpp { line[x - bpp] } else { 0 };
                    let up = prev_buf[x];
                    let ul = if x >= bpp { prev_buf[x - bpp] } else { 0 };
                    line[x] = scan[x].wrapping_add(unfilter_paeth(left, up, ul));
                }
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "unknown filter",
                ))
            }
        }
        // After processing, copy `line` into `prev_buf` for the next iteration's Up/Average/Paeth.
        prev_buf.copy_from_slice(line);
    }
    Ok((w, h, ctype, pixels))
}

pub fn decode_to_gray(r: &mut dyn Read) -> std::io::Result<GrayImage> {
    let (w, h, ctype, pixels) = read_png(r)?;
    match ctype {
        0 => Ok(GrayImage::from_vec(pixels, w, h)),
        2 => {
            // Convert RGB to gray on the fly.
            let mut gray = vec![0u8; w * h];
            for i in 0..(w * h) {
                let r = pixels[i * 3] as u32;
                let g = pixels[i * 3 + 1] as u32;
                let b = pixels[i * 3 + 2] as u32;
                gray[i] = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
            }
            Ok(GrayImage::from_vec(gray, w, h))
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unsupported color type for gray decode",
        )),
    }
}

pub fn decode_to_rgb(r: &mut dyn Read) -> std::io::Result<RgbImage> {
    let (w, h, ctype, pixels) = read_png(r)?;
    match ctype {
        2 => Ok(RgbImage {
            data: pixels,
            width: w,
            height: h,
        }),
        0 => {
            let mut rgb = vec![0u8; w * h * 3];
            for i in 0..(w * h) {
                let v = pixels[i];
                rgb[i * 3] = v;
                rgb[i * 3 + 1] = v;
                rgb[i * 3 + 2] = v;
            }
            Ok(RgbImage {
                data: rgb,
                width: w,
                height: h,
            })
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unsupported color type for rgb decode",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn png_gray_roundtrip() {
        let mut img = GrayImage::new(8, 6);
        for y in 0..6 {
            for x in 0..8 {
                img[(x, y)] = ((x * 31 + y * 17) & 0xFF) as u8;
            }
        }
        let mut buf = Vec::new();
        write_png_gray(&mut buf, &img).unwrap();
        let back = decode_to_gray(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(back.width(), 8);
        assert_eq!(back.height(), 6);
        for y in 0..6 {
            for x in 0..8 {
                assert_eq!(img[(x, y)], back[(x, y)]);
            }
        }
    }

    #[test]
    fn png_rgb_roundtrip() {
        let mut img = RgbImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                let row = img.row_mut(y);
                row[x * 3] = (x * 30) as u8;
                row[x * 3 + 1] = (y * 60) as u8;
                row[x * 3 + 2] = 128;
            }
        }
        let mut buf = Vec::new();
        write_png_rgb(&mut buf, &img).unwrap();
        let back = decode_to_rgb(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(back.width(), 4);
        assert_eq!(back.height(), 4);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(img.row(y)[x * 3], back.row(y)[x * 3]);
                assert_eq!(img.row(y)[x * 3 + 1], back.row(y)[x * 3 + 1]);
                assert_eq!(img.row(y)[x * 3 + 2], back.row(y)[x * 3 + 2]);
            }
        }
    }
}
