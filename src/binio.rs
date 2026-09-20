//! Little-endian binary cursor helpers shared by the gallery/model codecs
//! ([`crate::lbph_store`], [`crate::subspace_store`]).
//!
//! Hand-rolled so the default build needs no third-party serialization crate.
//! Multi-byte integers are always little-endian; `f32` moves as raw IEEE-754
//! bits, which makes codec round-trips bit-exact.

/// A fixed-width read ran past the end of the buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BinError {
    /// Fewer bytes remained than the field needed.
    Truncated,
}

/// Append a `u16` in little-endian.
pub(crate) fn push_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Append a `u32` in little-endian.
pub(crate) fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Append an IEEE-754 `f32` in little-endian (raw bits).
pub(crate) fn push_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Bounds-checked little-endian cursor over a borrowed byte slice.
pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    pub(crate) fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Consume exactly `n` bytes or fail with [`BinError::Truncated`].
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], BinError> {
        let end = self.pos.checked_add(n).ok_or(BinError::Truncated)?;
        let chunk = self.buf.get(self.pos..end).ok_or(BinError::Truncated)?;
        self.pos = end;
        Ok(chunk)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, BinError> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16, BinError> {
        let mut a = [0u8; 2];
        a.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(a))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, BinError> {
        let mut a = [0u8; 4];
        a.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(a))
    }

    pub(crate) fn f32(&mut self) -> Result<f32, BinError> {
        let mut a = [0u8; 4];
        a.copy_from_slice(self.take(4)?);
        Ok(f32::from_le_bytes(a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_and_detects_truncation() {
        let mut b = Vec::new();
        push_u16(&mut b, 0x1234);
        push_u32(&mut b, 0x0102_0304);
        push_f32(&mut b, 2.5);
        let mut r = Reader::new(&b);
        assert_eq!(r.u16(), Ok(0x1234));
        assert_eq!(r.u32(), Ok(0x0102_0304));
        assert_eq!(r.f32(), Ok(2.5));
        assert_eq!(r.remaining(), 0);
        assert_eq!(r.u8(), Err(BinError::Truncated));

        let mut short = Reader::new(&b[..3]);
        assert_eq!(short.u16(), Ok(0x1234));
        assert_eq!(short.u32(), Err(BinError::Truncated));
        assert_eq!(short.position(), 2); // cursor does not advance on failure
    }

    #[test]
    fn f32_bits_are_exact() {
        for v in [
            f32::MIN_POSITIVE,
            -0.0,
            1.0 / 3.0,
            f32::MAX,
            f32::NEG_INFINITY,
        ] {
            let mut b = Vec::new();
            push_f32(&mut b, v);
            let back = Reader::new(&b).f32().unwrap();
            assert_eq!(back.to_le_bytes(), v.to_le_bytes());
        }
    }
}
