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
    retry: RetryPolicy,
}

#[derive(Debug)]
pub struct S3Error(pub String);

impl std::fmt::Display for S3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "s3: {}", self.0)
    }
}
impl std::error::Error for S3Error {}

impl S3Error {
    /// 是否值得重试:5xx + transport 层(connect / read / write timeout)+ 429。
    /// 4xx 一律不重试(bad bucket / bad key / signature / forbidden)。
    /// 解析签名错误也不重试 — 重试只是浪费代币 + 加重服务端负担。
    pub fn is_transient(&self) -> bool {
        let s = &self.0;
        if s.starts_with("transport:") {
            return true;
        }
        // 格式: `status <code>: <message>` 或 `status <code> for <method> <path>`
        if let Some(rest) = s.strip_prefix("status ") {
            // 取第一个 token 作为 code
            let code_str = rest.split([':', ' ']).next().unwrap_or("");
            if let Ok(code) = code_str.parse::<u16>() {
                return code == 429 || (500..600).contains(&code);
            }
        }
        false
    }
}

/// 指数退避重试策略。`max_attempts` 包括首次失败的那一次。
/// 尝试序列:`base`, `base*2`, `base*4`, ... 上限 `max_delay`。
#[derive(Clone, Copy, Debug)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub base_delay: std::time::Duration,
    pub max_delay: std::time::Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            // 4 次尝试:首失败 + 3 次重试(累计 ~700ms / 1.5s / 3.1s,不超过 max_delay)。
            max_attempts: 4,
            base_delay: std::time::Duration::from_millis(100),
            max_delay: std::time::Duration::from_secs(2),
        }
    }
}

/// 用平台 env 变量构造:从 `Config::s3_max_retries` / `s3_retry_base_ms`
/// / `s3_retry_max_ms` 推断。供 `main.rs` 在创建 `S3Client` 时调。
impl RetryPolicy {
    pub fn from_env_counts(max_attempts: u32, base_ms: u64, max_ms: u64) -> Self {
        Self {
            max_attempts: max_attempts.max(1),
            base_delay: std::time::Duration::from_millis(base_ms),
            max_delay: std::time::Duration::from_millis(max_ms.max(base_ms)),
        }
    }
}

/// 同步 retry 包装:`op` 在瞬态失败时按指数退避重试,非瞬态立即返回。
/// `attempts_left = 0` 表示不再重试,直接退出;`op` 一次调用消耗一次。
fn retry_sync<T, F>(policy: RetryPolicy, mut op: F) -> Result<T, S3Error>
where
    F: FnMut() -> Result<T, S3Error>,
{
    let mut delay = policy.base_delay;
    let mut last_err: Option<S3Error> = None;
    for attempt in 0..policy.max_attempts {
        match op() {
            Ok(v) => {
                if attempt > 0 {
                    tracing::info!("[s3] succeeded after {} retry", attempt);
                }
                return Ok(v);
            }
            Err(e) => {
                let transient = e.is_transient();
                if !transient || attempt + 1 >= policy.max_attempts {
                    return Err(e);
                }
                tracing::warn!(
                    "[s3] transient error (attempt {}/{}): {} — sleeping {:?} before retry",
                    attempt + 1,
                    policy.max_attempts,
                    e,
                    delay
                );
                last_err = Some(e);
                std::thread::sleep(delay);
                // 退避:base * 2^attempt,夹到 max_delay。
                delay = std::cmp::min(delay.saturating_mul(2), policy.max_delay);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| S3Error("retry exhausted with no error".into())))
}

impl S3Client {
    #[allow(dead_code)] // 保留为 rsface_platform library API;bin 走 with_retry
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
            retry: RetryPolicy::default(),
        }
    }

    /// 用自定义重试策略构造(供测试 / 需要关闭重试的场景)。
    #[allow(dead_code)]
    pub fn with_retry(
        endpoint: String,
        region: String,
        access_key: String,
        secret_key: String,
        bucket: String,
        retry: RetryPolicy,
    ) -> Self {
        let endpoint = endpoint.trim_end_matches('/').to_string();
        Self {
            agent: agent_builder_no_tls(),
            endpoint,
            region,
            access_key,
            secret_key,
            bucket,
            retry,
        }
    }

    /// 桶不存在则创建;HEAD 失败区分 404 vs 403,避免坏凭据被当桶不存在再试一次。
    pub fn ensure_bucket(&self) -> Result<(), S3Error> {
        retry_sync(self.retry, || {
            match self.request("HEAD", "/", &[], &[], None) {
                Ok(_) => Ok(()),
                Err(S3Error(e))
                    if e.contains("404") || e.contains("NotFound") || e.contains("NoSuchKey") =>
                {
                    self.request("PUT", "/", &[], &[], None).map(|_| ())
                }
                Err(other) => Err(other),
            }
        })
    }

    pub fn put_object(&self, key: &str, content_type: &str, body: Vec<u8>) -> Result<(), S3Error> {
        retry_sync(self.retry, || {
            self.request(
                "PUT",
                &format!("/{key}"),
                &[],
                &[("content-type", content_type)],
                Some(Body::Bytes(body.clone())),
            )
            .map(|_| ())
        })
    }

    /// 流式 PUT:从 `path` 直接发送文件内容,body 在请求过程中按需 read。
    /// 用于大文件上传到 S3 时避免 `fs::read` + `to_vec` 双倍拷贝。
    /// 调用方负责文件存在性与大小;签名阶段 ureq 会 seek-to-end 读 Content-Length。
    ///
    /// 注:大文件流的 retry 不能简单重发整个文件 — 这里只对短耗时操作
    /// (open / metadata stat / 短包上传) 走 retry;长传输过程中失败留给上层
    /// 重传(put_with_fallback 已能落到 local 兜底)。
    pub fn put_object_file(
        &self,
        key: &str,
        content_type: &str,
        path: &std::path::Path,
    ) -> Result<(), S3Error> {
        // open + stat 是同步 fs 操作,失败非瞬态;在 retry 之外做一次。
        let f = std::fs::File::open(path)
            .map_err(|e| S3Error(format!("open for put {path:?}: {e}")))?;
        let len = f
            .metadata()
            .map_err(|e| S3Error(format!("stat for put {path:?}: {e}")))?
            .len();
        // 仅在网络层失败时重试 — 文件 fd 不可复制,重试需重新 open。
        retry_sync(self.retry, || {
            let f = std::fs::File::open(path)
                .map_err(|e| S3Error(format!("open for put {path:?}: {e}")))?;
            self.request(
                "PUT",
                &format!("/{key}"),
                &[],
                &[
                    ("content-type", content_type),
                    ("content-length", &len.to_string()),
                ],
                Some(Body::File(f)),
            )
            .map(|_| ())
        })
    }

    pub fn get_object(&self, key: &str) -> Result<(Vec<u8>, String), S3Error> {
        retry_sync(self.retry, || {
            let resp = self.request("GET", &format!("/{key}"), &[], &[], None)?;
            let ct = resp
                .header("content-type")
                .unwrap_or("application/octet-stream")
                .to_string();
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut resp.into_reader(), &mut buf)
                .map_err(|e| S3Error(e.to_string()))?;
            Ok((buf, ct))
        })
    }

    /// 带 Range 的 GET(S3 语义:`bytes=start-end`,end 含端,可省略表示到 EOF)。
    /// 返回 (切片字节, 服务端给的 total 长度)。
    /// 用于 compare_algos 的 bounded read (≤ 16 MiB 用 in-memory 即可,
    /// 避免 streaming 复杂度)。
    #[allow(dead_code)]
    pub fn get_object_range(
        &self,
        key: &str,
        start: u64,
        end: Option<u64>,
    ) -> Result<(Vec<u8>, u64), S3Error> {
        retry_sync(self.retry, || {
            let range = match end {
                Some(e) => format!("bytes={start}-{e}"),
                None => format!("bytes={start}-"),
            };
            let resp = self.request("GET", &format!("/{key}"), &[], &[("range", &range)], None)?;
            let cr_total = resp
                .header("content-range")
                .and_then(|v| v.rsplit('/').next())
                .and_then(|t| t.parse::<u64>().ok());
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut resp.into_reader(), &mut buf)
                .map_err(|e| S3Error(e.to_string()))?;
            let total = cr_total.unwrap_or(buf.len() as u64);
            Ok((buf, total))
        })
    }

    /// 流式 Range GET:返回 `tokio::io::AsyncRead` 适配器 + Content-Range total。
    /// Content-Length 头 + 同步读 → 异步适配,后台驱动 ureq 同步 socket)。
    /// 总字节数来自 Content-Range 的 total 字段。
    /// 高 #3:用于 /media 视频拖进度条,避免一次性 GB 级别 `to_vec`。
    pub fn get_object_range_stream(
        &self,
        key: &str,
        start: u64,
        end: Option<u64>,
    ) -> Result<(S3RangeStream, u64), S3Error> {
        retry_sync(self.retry, || {
            let range = match end {
                Some(e) => format!("bytes={start}-{e}"),
                None => format!("bytes={start}-"),
            };
            let resp = self.request("GET", &format!("/{key}"), &[], &[("range", &range)], None)?;
            let total = resp
                .header("content-range")
                .and_then(|v| v.rsplit('/').next())
                .and_then(|t| t.parse::<u64>().ok())
                .ok_or_else(|| S3Error("missing Content-Range total".into()))?;
            // resp.into_reader() 是 sync std::io::Read;包到 S3RangeStream。
            let sync_reader = resp.into_reader();
            Ok((S3RangeStream::new(sync_reader), total))
        })
    }

    /// HEAD object:取 Content-Length(不下载 body)。Range 请求需要先知道
    /// 对象总大小来构造 `Content-Range` 响应头。
    pub fn head_object(&self, key: &str) -> Result<u64, S3Error> {
        retry_sync(self.retry, || {
            let resp = self.request("HEAD", &format!("/{key}"), &[], &[], None)?;
            resp.header("content-length")
                .and_then(|v| v.parse::<u64>().ok())
                .ok_or_else(|| S3Error("HEAD missing content-length".into()))
        })
    }

    /// 轻量健康检查:`HEAD /<bucket>` 看 rustfs 是否可达 + 凭据是否合法。
    /// 用于 `/api/health/deep` 端点。
    pub fn ping(&self) -> Result<(), S3Error> {
        retry_sync(self.retry, || {
            self.request("HEAD", "/", &[], &[], None).map(|_| ())
        })
    }

    /// ListObjectsV2:列出 `prefix` 下全部对象 key(自动按 continuation-token
    /// 翻页)。删除任务媒体时用来枚举对象。
    pub fn list_objects(&self, prefix: &str) -> Result<Vec<String>, S3Error> {
        let mut keys = Vec::new();
        let mut token: Option<String> = None;
        loop {
            // 每页独立 retry:翻页中途失败不会让前面已收到的 keys 失效,
            // 整段失败重试也是同样的 query,服务端的 continuation-token
            // 在短期抖动后通常仍可用。
            let page = retry_sync(self.retry, || {
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
                let keys = extract_xml_tag_values(&xml, "Key");
                let truncated = extract_xml_tag_values(&xml, "IsTruncated")
                    .first()
                    .map(|s| s == "true")
                    .unwrap_or(false);
                let next = extract_xml_tag_values(&xml, "NextContinuationToken")
                    .into_iter()
                    .next();
                Ok((keys, truncated, next))
            })?;
            let (page_keys, truncated, next) = page;
            keys.extend(page_keys);
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
        retry_sync(self.retry, || {
            self.request("DELETE", &format!("/{key}"), &[], &[], None)
                .map(|_| ())
        })
    }

    // ---- 内部:签名 + 发请求 ----

    fn request(
        &self,
        method: &str,
        path: &str,
        query: &[(&str, &str)],
        extra_headers: &[(&str, &str)],
        body: Option<Body>,
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

        // body 哈希:小/未知 body 用真 SHA-256;流式 File body 用 S3 SigV4
        // 标准 `UNSIGNED-PAYLOAD` sentinel(S3 + rustfs 都支持),避免
        // `fs::read` 进内存来给 GB 级上传算 hash。
        let (payload_hash, body_kind) = match body.as_ref() {
            None => (hex(&sha2::Sha256::digest([])), BodyKind::Empty),
            Some(Body::Bytes(b)) => (hex(&sha2::Sha256::digest(b)), BodyKind::Bytes),
            Some(Body::File(_)) => ("UNSIGNED-PAYLOAD".to_string(), BodyKind::File),
        };
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

        let result = match (body_kind, body) {
            (BodyKind::Empty, _) => req.call(),
            (BodyKind::Bytes, Some(Body::Bytes(b))) => req.send_bytes(&b),
            (BodyKind::File, Some(Body::File(f))) => req.send(f),
            _ => req.call(),
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

/// S3 请求 body:内存字节或磁盘文件(后者走 UNSIGNED-PAYLOAD 签名,
/// body 在传输阶段按需 read,不上传整文件到内存)。
pub enum Body {
    Bytes(Vec<u8>),
    File(std::fs::File),
}

#[derive(Clone, Copy)]
enum BodyKind {
    Empty,
    Bytes,
    File,
}

/// 同步 std::io::Read → tokio::io::AsyncRead 适配器。
///
/// 设计:后台线程同步 read,然后通过 `tokio::sync::mpsc` channel 把每段
/// bytes 推给前端 poll_read;只缓存一段在内存,GB 级视频不会驻留堆。
/// 用于把 ureq 的同步响应体桥到 axum 的流式 Body,避免大文件
/// `read_to_end` + `to_vec` 双倍拷贝。
pub struct S3RangeStream {
    rx: tokio::sync::mpsc::Receiver<std::io::Result<Vec<u8>>>,
    /// 当前持有的 bytes(上一段还没消费完的部分)。
    pending: std::io::Cursor<Vec<u8>>,
}

impl S3RangeStream {
    pub fn new(r: Box<dyn std::io::Read + Send>) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel::<std::io::Result<Vec<u8>>>(1);
        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(move || {
                use std::io::Read;
                let mut r = r;
                let mut buf = vec![0u8; 64 * 1024];
                loop {
                    match r.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            if tx.blocking_send(Ok(buf[..n].to_vec())).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx.blocking_send(Err(e));
                            break;
                        }
                    }
                }
            })
            .expect("spawn s3 stream pump");
        Self {
            rx,
            pending: std::io::Cursor::new(Vec::new()),
        }
    }
}

impl tokio::io::AsyncRead for S3RangeStream {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        loop {
            // 已有 pending 字节:消费一部分。
            let pos = self.pending.position() as usize;
            let len = self.pending.get_ref().len();
            if pos < len {
                let remaining = &self.pending.get_ref()[pos..];
                let n = remaining.len().min(buf.remaining());
                buf.put_slice(&remaining[..n]);
                self.pending.set_position((pos + n) as u64);
                return std::task::Poll::Ready(Ok(()));
            }
            // 没有 pending,等下一段。
            match self.rx.poll_recv(cx) {
                std::task::Poll::Ready(Some(Ok(bytes))) => {
                    self.pending = std::io::Cursor::new(bytes);
                }
                std::task::Poll::Ready(Some(Err(e))) => return std::task::Poll::Ready(Err(e)),
                std::task::Poll::Ready(None) => return std::task::Poll::Ready(Ok(())),
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}

fn agent_builder_no_tls() -> Agent {
    // 低 #15:原来 `.timeout(120s)` 是整个请求的全局上限,大对象 GET 会
    // 在 read 阶段被卡死。拆成 connect / read:connect 短(网络层异常),
    // read 长(允许 GB 级流式拉取)。
    ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(10))
        .timeout_read(std::time::Duration::from_secs(600))
        .timeout_write(std::time::Duration::from_secs(600))
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
    fn s3_error_classifies_transient() {
        // 5xx + 429 = 瞬态
        assert!(S3Error("status 500: boom".into()).is_transient());
        assert!(S3Error("status 503: service unavailable".into()).is_transient());
        assert!(S3Error("status 429: rate limit".into()).is_transient());
        // transport 层
        assert!(S3Error("transport: connection refused".into()).is_transient());
        // 4xx = 非瞬态
        assert!(!S3Error("status 400: bad request".into()).is_transient());
        assert!(!S3Error("status 403: forbidden".into()).is_transient());
        assert!(!S3Error("status 404: not found".into()).is_transient());
        assert!(!S3Error("status 416: range not satisfiable".into()).is_transient());
        // 未识别格式 = 非瞬态(安全侧:宁可漏报也不误重试)
        assert!(!S3Error("random parse failure".into()).is_transient());
    }

    #[test]
    fn retry_sync_succeeds_on_second_attempt() {
        // 首失败(瞬态) → 二成功 → 返回 Ok
        let mut calls = 0u32;
        let policy = RetryPolicy {
            max_attempts: 4,
            base_delay: std::time::Duration::from_millis(1),
            max_delay: std::time::Duration::from_millis(2),
        };
        let res: Result<u32, S3Error> = retry_sync(policy, || {
            calls += 1;
            if calls == 1 {
                Err(S3Error("status 503: try later".into()))
            } else {
                Ok(42)
            }
        });
        assert_eq!(res.unwrap(), 42);
        assert_eq!(calls, 2);
    }

    #[test]
    fn retry_sync_gives_up_on_non_transient() {
        // 4xx 立即放弃,不重试
        let mut calls = 0u32;
        let policy = RetryPolicy {
            max_attempts: 4,
            base_delay: std::time::Duration::from_millis(1),
            max_delay: std::time::Duration::from_millis(2),
        };
        let res: Result<(), S3Error> = retry_sync(policy, || {
            calls += 1;
            Err(S3Error("status 404: not found".into()))
        });
        assert!(res.is_err());
        assert_eq!(calls, 1);
    }

    #[test]
    fn retry_sync_exhausts_attempts() {
        // 持续 503:4 次尝试都用完,然后返回最后一次错误
        let mut calls = 0u32;
        let policy = RetryPolicy {
            max_attempts: 3,
            base_delay: std::time::Duration::from_millis(1),
            max_delay: std::time::Duration::from_millis(2),
        };
        let res: Result<(), S3Error> = retry_sync(policy, || {
            calls += 1;
            Err(S3Error("status 500: persistent".into()))
        });
        assert!(res.is_err());
        assert_eq!(calls, 3);
    }

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
