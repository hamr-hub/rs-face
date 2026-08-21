//! HTTP image source — single PNG, or image sequence over HTTP.
//!
//! Uses the `std::net::TcpStream` plus a hand-rolled HTTP/1.1 client so we have
//! zero external deps.

use super::{Frame, FrameSource};
use crate::image::png;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

pub struct HttpImageSource {
    kind: HttpKind,
    base: String,
    host: String,
    path_prefix: String,
    is_tls: bool,
    frame_index: u64,
    total_hint: Option<u64>,
}

enum HttpKind {
    Single,
    Sequence { width: u32, count: u64 },
}

impl HttpImageSource {
    pub fn new_single(url: &str) -> Self {
        let (is_tls, host, path) = parse_url(url);
        Self {
            kind: HttpKind::Single,
            base: url.to_string(),
            host,
            path_prefix: path,
            is_tls,
            frame_index: 0,
            total_hint: Some(1),
        }
    }

    pub fn new_sequence(base: &str, count: u64) -> Self {
        let (is_tls, host, path) = parse_url(base);
        Self {
            kind: HttpKind::Sequence { width: 5, count },
            base: base.to_string(),
            host,
            path_prefix: path,
            is_tls,
            frame_index: 0,
            total_hint: Some(count),
        }
    }
}

impl FrameSource for HttpImageSource {
    fn next_frame(&mut self) -> io::Result<Option<Frame>> {
        match self.kind {
            HttpKind::Single => {
                if self.frame_index > 0 {
                    return Ok(None);
                }
                let stream = open_http_get(&self.host, &self.path_prefix, self.is_tls)?;
                let (status, _headers, mut body) = read_http_response(stream)?;
                if status != 200 {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("HTTP {}", status),
                    ));
                }
                let rgb = png::decode_to_rgb(&mut body)?;
                let gray = rgb.to_gray();
                self.frame_index += 1;
                Ok(Some(Frame {
                    index: 0,
                    timestamp_ms: 0,
                    gray: Arc::new(gray),
                    rgb: Some(Arc::new(rgb)),
                }))
            }
            HttpKind::Sequence { width, count } => {
                if self.frame_index >= count {
                    return Ok(None);
                }
                let url = format!(
                    "{}frame_{:0width$}.png",
                    self.base,
                    self.frame_index,
                    width = width as usize
                );
                let (_tls, host_port, path) = parse_url(&url);
                let stream = open_http_get(&host_port, &path, _tls)?;
                let (status, _headers, mut body) = read_http_response(stream)?;
                if status != 200 {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("HTTP {} for {}", status, url),
                    ));
                }
                let gray = png::decode_to_gray(&mut body)?;
                let idx = self.frame_index;
                self.frame_index += 1;
                Ok(Some(Frame {
                    index: idx,
                    timestamp_ms: idx * 33,
                    gray: Arc::new(gray),
                    rgb: None,
                }))
            }
        }
    }

    fn total_hint(&self) -> Option<u64> {
        self.total_hint
    }
}

pub(crate) fn parse_url(url: &str) -> (bool, String, String) {
    let (scheme, rest) = if let Some(pos) = url.find("://") {
        let s = &url[..pos];
        let rest = &url[pos + 3..];
        (s.to_string(), rest.to_string())
    } else {
        ("http".to_string(), url.to_string())
    };
    let is_tls = scheme == "https";
    let (host_port, path) = match rest.find('/') {
        Some(idx) => (rest[..idx].to_string(), rest[idx..].to_string()),
        None => (rest.clone(), "/".to_string()),
    };
    (is_tls, host_port, path)
}

fn split_host_port(host_port: &str) -> (String, Option<u16>) {
    if let Some(idx) = host_port.rfind(':') {
        let host = host_port[..idx].to_string();
        let port = host_port[idx + 1..].parse::<u16>().ok();
        (host, port)
    } else {
        (host_port.to_string(), None)
    }
}

pub(crate) fn open_http_get(host_port: &str, path: &str, _is_tls: bool) -> io::Result<TcpStream> {
    let (host, port) = split_host_port(host_port);
    let port = port.unwrap_or(80);
    let stream = TcpStream::connect(format!("{}:{}", host, port))?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    let mut s = stream;
    write!(s, "GET {} HTTP/1.1\r\nHost: {}\r\nUser-Agent: rs-face/0.1\r\nAccept: image/png,image/*\r\nConnection: close\r\n\r\n", path, host_port)?;
    s.flush()?;
    Ok(s)
}

pub(crate) fn read_http_response(
    stream: TcpStream,
) -> io::Result<(u16, Vec<(String, String)>, HttpBody)> {
    let mut br = BufReader::new(stream);
    let mut status_line = String::new();
    br.read_line(&mut status_line)?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad status"))?;
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        let n = br.read_line(&mut line)?;
        if n == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some(idx) = line.find(':') {
            let k = line[..idx].trim().to_string();
            let v = line[idx + 1..].trim().to_string();
            headers.push((k, v));
        }
    }
    let body = HttpBody { inner: br };
    Ok((status, headers, body))
}

/// Minimal HTTP body reader.
pub(crate) struct HttpBody {
    inner: BufReader<TcpStream>,
}

impl Read for HttpBody {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}
