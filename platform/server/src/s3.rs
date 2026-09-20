//! 极简 S3 兼容客户端(AWS SigV4),面向 rustfs。
//!
//! 只实现平台需要的操作:`put_object` / `get_object` / `ensure_bucket`。
//! 键字符集约定为 `[A-Za-z0-9/._-]`,避免完整 URI 编码。

use hmac::{Hmac, Mac};
use sha2::Digest;
use ureq::Agent;

#[derive(Clone)]
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
        match self.request("HEAD", "/", &[], &[], None) {
            Ok(_) => Ok(()),
            Err(_) => self.request("PUT", "/", &[], &[], None).map(|_| ()),
        }
    }

    pub fn put_object(&self, key: &str, content_type: &str, body: Vec<u8>) -> Result<(), S3Error> {
        let body = Some(body);
        self.request(
            "PUT",
            &format!("/{key}"),
            &[],
            &[("content-type", content_type)],
            body,
        )
        .map(|_| ())
    }

    pub fn get_object(&self, key: &str) -> Result<(Vec<u8>, String), S3Error> {
        let resp = self.request("GET", &format!("/{key}"), &[], &[], None)?;
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
        let resp = self.request("GET", &format!("/{key}"), &[], &[("range", &range)], None)?;
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
        let resp = self.request("HEAD", &format!("/{key}"), &[], &[], None)?;
        resp.header("content-length")
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or_else(|| S3Error("HEAD missing content-length".into()))
    }

    /// ListObjectsV2:列出 `prefix` 下全部对象 key(自动按 continuation-token
    /// 翻页)。删除任务媒体时用来枚举对象。
    pub fn list_objects(&self, prefix: &str) -> Result<Vec<String>, S3Error> {
        let mut keys = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let mut query: Vec<(String, String)> = vec![
                ("list-type".to_string(), "2".to_string()),
                ("prefix".to_string(), prefix.to_string()),
            ];
            if let Some(t) = &token {
                query.push(("continuation-token".to_string(), t.clone()));
            }
            let q_ref: Vec<(&str, &str)> = query
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            let resp = self.request("GET", "/", &q_ref, &[], None)?;
            let mut xml = String::new();
            std::io::Read::read_to_string(&mut resp.into_reader(), &mut xml)
                .map_err(|e| S3Error(e.to_string()))?;
            keys.extend(extract_xml_tag_values(&xml, "Key"));
            let truncated = extract_xml_tag_values(&xml, "IsTruncated")
                .first()
                .map(|s| s == "true")
                .unwrap_or(false);
            let next = extract_xml_tag_values(&xml, "NextContinuationToken")
                .into_iter()
                .next();
            match (truncated, next) {
                (true, Some(t)) => token = Some(t),
                _ => break,
            }
        }
        Ok(keys)
    }

    /// DELETE object。任务删除时逐个调用(平台是 LAN 单桶,按键 DELETE 无需
    /// DeleteObjects 必需的 Content-MD5,免引 md5 实现)。
    pub fn delete_object(&self, key: &str) -> Result<(), S3Error> {
        self.request("DELETE", &format!("/{key}"), &[], &[], None)
            .map(|_| ())
    }

    // ---- 内部:签名 + 发请求 ----

    fn request(
        &self,
        method: &str,
        path: &str,
        query: &[(&str, &str)],
        extra_headers: &[(&str, &str)],
        body: Option<Vec<u8>>,
    ) -> Result<ureq::Response, S3Error> {
        let path_trimmed = path.trim_start_matches('/');
        // path 风格 /bucket/key(virtual-host 风格不适用 rustfs 单 IP 部署)。
        let url_base = format!(
            "{}/{}/{}",
            self.endpoint,
            url_encode_path_segment(&self.bucket),
            path_trimmed
        );
        // 供签名用的 path。
        let sign_path = format!(
            "/{}/{}",
            url_encode_path_segment(&self.bucket),
            path_trimmed
        );

        // canonical query:按 key(再 value)排序后 RFC3986 编码。
        // 旧实现把 query 直接拼进 sign_path 却签成空 query,带参请求会
        // SignatureDoesNotMatch;list_objects 依赖这里签名正确。
        let mut query_sorted: Vec<(&str, &str)> = query.to_vec();
        query_sorted.sort_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(b.1)));
        let canonical_query: String = query_sorted
            .iter()
            .map(|(k, v)| {
                format!(
                    "{}={}",
                    url_encode_query_component(k),
                    url_encode_query_component(v)
                )
            })
            .collect::<Vec<_>>()
            .join("&");
        // host 取自 base(query 里的 '/' 已全编码为 %2F,但先取更直接)。
        let host = host_of(&url_base);
        let url = if canonical_query.is_empty() {
            url_base
        } else {
            format!("{url_base}?{canonical_query}")
        };

        let body_bytes = body.unwrap_or_default();
        let payload_hash = hex(&sha2::Sha256::digest(&body_bytes));
        let amz_date = now_amz_date();

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
            "{method}\n{sign_path}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
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
            .set("Host", &host)
            .set("x-amz-date", &amz_date)
            .set("x-amz-content-sha256", &payload_hash);
        // extra_headers 必须完整发出 —— 签名时它们已经在 canonical_headers 里,
        // 实际请求漏发任何一项都会触发 SignatureDoesNotMatch。
        for (k, v) in extra_headers {
            req = req.set(k, v);
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
fn url_encode_path_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// query key/value 编码:除 unreserved 字符外全部百分号编码,`/` 也编码
/// 为 %2F(S3 对 query 与 path 的编码规则不同;若留下裸 `/`,服务端重算
/// 签名时会再编码,导致 SignatureDoesNotMatch)。
fn url_encode_query_component(s: &str) -> String {
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

/// 从 S3 ListObjectsV2 的 XML 响应里取出指定标签的文本值。
/// 不引 XML 解析器:响应结构简单且标签无嵌套同名;对象 key 字符集受限,
/// 只处理最基本的 `&amp;` `&lt;` `&gt;` 实体。
fn extract_xml_tag_values(xml: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find(&open) {
        let after = &rest[start + open.len()..];
        match after.find(&close) {
            Some(end) => {
                out.push(xml_unescape(&after[..end]));
                rest = &after[end + close.len()..];
            }
            None => break,
        }
    }
    out
}

fn xml_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
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

    #[test]
    fn host_of_keeps_port_and_strips_path() {
        // SigV4 canonical `host` must equal the authority actually sent,
        // non-default port included — this is what the Host-header fix signs.
        assert_eq!(
            host_of("http://127.0.0.1:9000/bucket/key"),
            "127.0.0.1:9000"
        );
        assert_eq!(host_of("https://s3.example.com/b/k?x=1"), "s3.example.com");
        assert_eq!(host_of("http://rustfs:9000/"), "rustfs:9000");
        // no scheme fallback: whole string up to the first '/'
        assert_eq!(host_of("localhost:9000/x"), "localhost:9000");
    }

    #[test]
    fn query_component_encodes_slash() {
        // prefix 里的 / 必须编码成 %2F,否则与服务端重算的签名不一致。
        assert_eq!(url_encode_query_component("jobs/a/b/"), "jobs%2Fa%2Fb%2F");
        assert_eq!(url_encode_query_component("a-b_c.d~1"), "a-b_c.d~1");
    }

    #[test]
    fn list_xml_keys_and_truncation() {
        let xml = "<?xml version=\"1.0\"?><ListBucketResult><Name>b</Name>\
<Contents><Key>jobs/1/original.mp4</Key></Contents>\
<Contents><Key>jobs/1/000001.png</Key></Contents>\
<IsTruncated>true</IsTruncated>\
<NextContinuationToken>abc123</NextContinuationToken></ListBucketResult>";
        let keys = extract_xml_tag_values(xml, "Key");
        assert_eq!(keys, vec!["jobs/1/original.mp4", "jobs/1/000001.png"]);
        assert_eq!(extract_xml_tag_values(xml, "IsTruncated"), vec!["true"]);
        assert_eq!(
            extract_xml_tag_values(xml, "NextContinuationToken"),
            vec!["abc123"]
        );
        // 不存在的标签返回空,不报错。
        assert!(extract_xml_tag_values(xml, "NoSuchTag").is_empty());
    }
}
