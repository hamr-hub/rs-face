//! 极简 S3 兼容客户端(AWS SigV4),面向 rustfs。
//!
//! 只实现平台需要的操作:`put_object` / `get_object` / `ensure_bucket`。
//! 键字符集约定为 `[A-Za-z0-9/._-]`,避免完整 URI 编码。

use hmac::{Hmac, Mac};
use sha2::Digest;
use ureq::Agent;

pub struct S3Client {
    agent: Agent,
    endpoint: String, // http://host:port,无尾斜杠
    region: String,
    access_key: String,
    secret_key: String,
    bucket: String,
}

#[derive(Debug)]
pub struct S3Error(pub String);

impl std::fmt::Display for S3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "s3: {}", self.0)
    }
}
impl std::error::Error for S3Error {}

impl S3Client {
    pub fn new(
        endpoint: String,
        region: String,
        access_key: String,
        secret_key: String,
        bucket: String,
    ) -> Self {
        let endpoint = endpoint.trim_end_matches('/').to_string();
        Self {
            agent: agent_builder_no_tls(),
            endpoint,
            region,
            access_key,
            secret_key,
            bucket,
        }
    }

    /// 桶不存在则创建;存在则跳过。
    pub fn ensure_bucket(&self) -> Result<(), S3Error> {
        match self.request("HEAD", "/", &[], None) {
            Ok(_) => Ok(()),
            Err(_) => self.request("PUT", "/", &[], None).map(|_| ()),
        }
    }

    pub fn put_object(&self, key: &str, content_type: &str, body: Vec<u8>) -> Result<(), S3Error> {
        let body = Some(body);
        self.request(
            "PUT",
            &format!("/{key}"),
            &[("content-type", content_type)],
            body,
        )
        .map(|_| ())
    }

    pub fn get_object(&self, key: &str) -> Result<(Vec<u8>, String), S3Error> {
        let resp = self.request("GET", &format!("/{key}"), &[], None)?;
        let ct = resp
            .header("content-type")
            .unwrap_or("application/octet-stream")
            .to_string();
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut resp.into_reader(), &mut buf)
            .map_err(|e| S3Error(e.to_string()))?;
        Ok((buf, ct))
    }

    /// 带 Range 的 GET(S3 语义:`bytes=start-end`,end 含端,可省略表示到 EOF)。
    /// 返回 (切片字节, 服务端给的 total 长度)。
    /// total 从 `Content-Range: bytes s-e/total` 解析;无该头时退化为 bytes.len()。
    /// 用于 /media 视频拖进度条(浏览器发 Range 请求)。
    pub fn get_object_range(
        &self,
        key: &str,
        start: u64,
        end: Option<u64>,
    ) -> Result<(Vec<u8>, u64), S3Error> {
        let range = match end {
            Some(e) => format!("bytes={start}-{e}"),
            None => format!("bytes={start}-"),
        };
        // range 作为 header 传给 request();签名时一并纳入 canonical headers。
        let resp = self.request("GET", &format!("/{key}"), &[("range", &range)], None)?;
        // 先取 header 再消费 reader(into_reader 之后 resp 已 move)。
        let cr_total = resp
            .header("content-range")
            .and_then(|v| v.rsplit('/').next())
            .and_then(|t| t.parse::<u64>().ok());
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut resp.into_reader(), &mut buf)
            .map_err(|e| S3Error(e.to_string()))?;
        let total = cr_total.unwrap_or(buf.len() as u64);
        Ok((buf, total))
    }

    /// HEAD object:取 Content-Length(不下载 body)。Range 请求需要先知道
    /// 对象总大小来构造 `Content-Range` 响应头。
    pub fn head_object(&self, key: &str) -> Result<u64, S3Error> {
        let resp = self.request("HEAD", &format!("/{key}"), &[], None)?;
        resp.header("content-length")
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or_else(|| S3Error("HEAD missing content-length".into()))
    }

    // ---- 内部:签名 + 发请求 ----

    fn request(
        &self,
        method: &str,
        path_and_query: &str,
        extra_headers: &[(&str, &str)],
        body: Option<Vec<u8>>,
    ) -> Result<ureq::Response, S3Error> {
        // path 与 bucket:virtual-host 风格用子域名,这里统一用 path 风格 /bucket/key。
        let url = format!(
            "{}/{}/{}",
            self.endpoint,
            url_encode_segment(&self.bucket),
            path_and_query.trim_start_matches('/')
        );
        // 供签名用的 path
        let sign_path = format!(
            "/{}/{}",
            url_encode_segment(&self.bucket),
            path_and_query.trim_start_matches('/')
        );

        let body_bytes = body.unwrap_or_default();
        let payload_hash = hex(&sha2::Sha256::digest(&body_bytes));
        let amz_date = now_amz_date();
        let host = host_of(&url);

        // canonical headers:content-type(可选), host, x-amz-content-sha256, x-amz-date,
        // range(可选,GET 部分 content 时必须与实际发送一致)
        let mut headers: Vec<(String, String)> = Vec::new();
        if let Some(ct) = extra_headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
        {
            headers.push(("content-type".to_string(), ct.1.to_string()));
        }
        if let Some(range) = extra_headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("range"))
        {
            headers.push(("range".to_string(), range.1.to_string()));
        }
        headers.push(("host".to_string(), host.clone()));
        headers.push(("x-amz-content-sha256".to_string(), payload_hash.clone()));
        headers.push(("x-amz-date".to_string(), amz_date.clone()));
        headers.sort_by(|a, b| a.0.cmp(&b.0));

        let canonical_headers: String = headers.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
        let signed_headers: String = headers
            .iter()
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>()
            .join(";");

        let canonical_request = format!(
            "{method}\n{sign_path}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        );

        let scope = format!("{}/{}/s3/aws4_request", &amz_date[..8], self.region);
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
            hex(&sha2::Sha256::digest(canonical_request.as_bytes()))
        );

        let signing_key = hmac_sha256(
            &hmac_sha256(
                &hmac_sha256(
                    &hmac_sha256(
                        format!("AWS4{}", self.secret_key).as_bytes(),
                        &amz_date.as_bytes()[..8],
                    )?,
                    self.region.as_bytes(),
                )?,
                b"s3",
            )?,
            b"aws4_request",
        )?;
        let signature = hex(&hmac_sha256_fixed(&signing_key, string_to_sign.as_bytes())?);

        let authorization = format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
            self.access_key);

        let mut req = self
            .agent
            .request(method, &url)
            .set("Authorization", &authorization)
            .set("x-amz-date", &amz_date)
            .set("x-amz-content-sha256", &payload_hash);
        for (k, v) in extra_headers {
            if !k.eq_ignore_ascii_case("content-type") {
                req = req.set(k, v);
            }
        }

        let result = if body_bytes.is_empty() && method != "PUT" {
            req.call()
        } else {
            req.send_bytes(&body_bytes)
        };
        match result {
            Ok(resp) => Ok(resp),
            Err(ureq::Error::Status(_code, resp)) => {
                // 4xx/5xx:S3 层错误以业务错误返回(HEAD 时 body 为空)。
                let mut msg = String::new();
                let _ = std::io::Read::read_to_string(&mut resp.into_reader(), &mut msg);
                if msg.is_empty() {
                    // HEAD 类无 body,用状态码描述
                    Err(S3Error(format!(
                        "status {} for {method} {}",
                        _code, sign_path
                    )))
                } else {
                    // 幂等场景:桶已存在等
                    if _code == 409
                        && method == "PUT"
                        && sign_path.ends_with(&format!("/{}", self.bucket))
                    {
                        Ok(ureq::Response::new(200, "OK", "").unwrap())
                    } else {
                        Err(S3Error(format!(
                            "status {}: {}",
                            _code,
                            truncate(&msg, 300)
                        )))
                    }
                }
            }
            Err(e) => Err(S3Error(format!("transport: {e}"))),
        }
    }
}

fn agent_builder_no_tls() -> Agent {
    ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(120))
        .build()
}

fn truncate(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

fn host_of(url: &str) -> String {
    let rest = url.split("://").nth(1).unwrap_or(url);
    rest.split('/').next().unwrap_or(rest).to_string()
}

/// path 风格地址段编码:仅编码不安全字符,`/` 由调用方控制。
fn url_encode_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Result<Vec<u8>, S3Error> {
    hmac_sha256_fixed(key, data)
}

fn hmac_sha256_fixed(key: &[u8], data: &[u8]) -> Result<Vec<u8>, S3Error> {
    let mut mac =
        Hmac::<sha2::Sha256>::new_from_slice(key).map_err(|e| S3Error(format!("hmac: {e}")))?;
    mac.update(data);
    Ok(mac.finalize().into_bytes().to_vec())
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// UTC 时间戳,格式 YYYYMMDD'T'HHMMSS'Z'。
fn now_amz_date() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    amz_date_from_unix(secs)
}

fn amz_date_from_unix(secs: u64) -> String {
    // 简化 civil-from-days 算法(Howard Hinnant)。
    let days = (secs / 86400) as i64;
    let rem = secs % 86400;
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}{m:02}{d:02}T{h:02}{mi:02}{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amz_date_shape() {
        // 2026-08-20 00:00:00 UTC == 1787184000
        // (修复:原常量 1787155200 实为 2026-08-19T16:00:00Z,断言错误)
        assert_eq!(amz_date_from_unix(1787184000), "20260820T000000Z");
        assert_eq!(amz_date_from_unix(0), "19700101T000000Z");
        // 闰年边界:2024-02-29 23:59:59 UTC == 1709251199
        assert_eq!(amz_date_from_unix(1709251199), "20240229T235959Z");
        // 年边界:2025-01-01 00:00:00 UTC == 1735689600
        assert_eq!(amz_date_from_unix(1735689600), "20250101T000000Z");
    }

    #[test]
    fn hex_shape() {
        assert_eq!(hex(&[0xde, 0xad]), "dead");
    }
}
