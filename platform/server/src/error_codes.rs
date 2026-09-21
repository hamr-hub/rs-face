//! 平台错误码枚举(2026-09-20 stability-hardening-2 加)。
//!
//! 目标:100% 覆盖 4xx/5xx 响应中的 `error_code` 字段,避免再出现 `internal` 这种
//! 无信息量的默认值。所有 handler 都从本模块取常量,而不是散落字符串字面量。
//!
//! 设计取舍:
//! - 用 `pub const &str` 而不是 `enum + as_str()`:常量更易在 `serde_json::json!`
//!   宏里直接拼,匹配 `match` 时也无需 `.as_str()`;
//! - 命名空间分层(prefix 域),便于 grep/日志过滤:`cascade_*` `pg_*` `s3_*`
//!   `decode_*` `job_*` `param_*` `auth_*` `media_*` `internal_*` `rate_*`;
//! - `INTERNAL` 是"我不知道"的兜底,不应该出现在新代码里 — 出现等于漏写。
//! - `unknown()` 对应 HTTP 404 "未分类" — 不再用 `internal` 兜底;
//! - 测试固定码表 + `assert_code_shape` 防御"snake_case + ASCII + ≤ 32"被违反,
//!   防止生产 API 出非机器可读字段。

/// 兜底码 — 新代码不允许出现,只给老 `error_response(msg)` 路径用。
pub const INTERNAL: &str = "internal";
/// HTTP 404 默认 — 缺省错误归类。
pub const UNKNOWN: &str = "unknown";

// ───── detector / cascade ──────────────────────────────────────────
/// Haar cascade 文件不存在(`cascade.rfcf` / `haarcascade_frontalface_default.xml`)。
pub const CASCADE_MISSING: &str = "cascade_missing";
/// Haar cascade 文件存在但解析失败(magic / version / 结构损坏)。
pub const CASCADE_CORRUPT: &str = "cascade_corrupt";
/// Haar cascade 加载失败但 reason 不能被归为 missing/corrupt。
pub const CASCADE_LOAD_FAILED: &str = "cascade_load_failed";
/// 用户/环境请求了不支持的算法名。
pub const UNSUPPORTED_ALGO: &str = "unsupported_algo";

// ───── postgres ────────────────────────────────────────────────────
/// PG 连接失败 / 连接池枯竭。
pub const PG_UNAVAILABLE: &str = "pg_unavailable";
/// PG 查询/写入超过 acquire_timeout(5s)。
pub const PG_TIMEOUT: &str = "pg_timeout";
/// UNIQUE / FK / CHECK 触发。
pub const PG_CONSTRAINT_VIOLATION: &str = "pg_constraint_violation";

// ───── s3 / rustfs ─────────────────────────────────────────────────
/// rustfs 不可达(连接失败 / DNS / 5xx)。
pub const S3_UNAVAILABLE: &str = "s3_unavailable";
/// SigV4 签名错(credential 错或 client/server 时钟漂移)。
pub const S3_SIGNATURE_MISMATCH: &str = "s3_signature_mismatch";
/// 桶不存在且 PUT 创建失败。
pub const S3_BUCKET_MISSING: &str = "s3_bucket_missing";
/// 对象不存在(GET / HEAD 返 404)。
pub const S3_OBJECT_NOT_FOUND: &str = "s3_object_not_found";

// ───── media decode ────────────────────────────────────────────────
/// 上传/导入的文件类型不被支持(不是 PNG/JPG/PGM/PPM/MP4/WEBM/MOV/MKV)。
pub const DECODE_UNSUPPORTED: &str = "decode_unsupported";
/// 文件被截断(Early EOF)。
pub const DECODE_TRUNCATED: &str = "decode_truncated";
/// 文件结构损坏(magic / CRC / chunk 错)。
pub const DECODE_CORRUPT: &str = "decode_corrupt";
/// ffmpeg 转码超时(单图 30s / 视频默认超时)。
pub const DECODE_TIMEOUT: &str = "decode_timeout";

// ───── job lifecycle ──────────────────────────────────────────────
/// job id 不在内存索引里(可能已被删除或属于另一实例)。
pub const JOB_NOT_FOUND: &str = "job_not_found";
/// job 已进入终态(Done/Cancelled/Error),不能再次 cancel / retry / 修改。
pub const JOB_ALREADY_DONE: &str = "job_already_done";
/// job 已被用户取消。
pub const JOB_CANCELLED: &str = "job_cancelled";
/// job 因为 worker panic 终止(隔离态,正常路径不该到这里)。
pub const JOB_PANIC: &str = "job_panic";

// ───── params / input ─────────────────────────────────────────────
/// 客户端发的参数在合法范围外(如 limit 负数 / offset 越界)。
pub const PARAM_OUT_OF_RANGE: &str = "param_out_of_range";
/// 必填字段缺失(multipart `file` 缺失 / JSON 字段为空)。
pub const PARAM_MISSING: &str = "param_missing";
/// 字段类型 / 格式错(parse 失败、URL scheme 不允许、host 空)。
pub const PARAM_INVALID: &str = "param_invalid";
/// 客户端批量请求的元素数超上限。
pub const PARAM_BATCH_TOO_LARGE: &str = "param_batch_too_large";

// ───── auth / security ────────────────────────────────────────────
/// 缺失 Authorization 头 / cookie(平台目前未启用,留位)。
pub const AUTH_MISSING: &str = "auth_missing";
/// token 错 / 签名失败。
pub const AUTH_INVALID: &str = "auth_invalid";
/// token TTL 过期。
pub const AUTH_EXPIRED: &str = "auth_expired";
/// 导入 URL 命中 SSRF 黑名单(loopback / metadata / IPv6 link-local)。
pub const SSRF_BLOCKED: &str = "ssrf_blocked";

// ───── media proxy ────────────────────────────────────────────────
/// 用户给的 key 含 `..` / 绝对路径 / 跳出 media 根。
pub const MEDIA_BAD_KEY: &str = "media_bad_key";
/// inline:// 误用 /media 路由(SSE 才支持)。
pub const MEDIA_GONE: &str = "media_gone";
/// Range 头合法但切片失败(文件被并发删除 / 截断)。
pub const MEDIA_RANGE_INVALID: &str = "media_range_invalid";
/// Range 不可满足(start >= total)。
pub const MEDIA_RANGE_UNSATISFIABLE: &str = "media_range_unsatisfiable";

// ───── upload ─────────────────────────────────────────────────────
/// multipart `file` 字段超过配置上限(`upload_limit_image/video`)。
pub const UPLOAD_TOO_LARGE: &str = "upload_too_large";
/// 0 字节上传(客户端断流 / multipart 没 part)。
pub const UPLOAD_EMPTY: &str = "upload_empty";
/// ffmpeg 准备原始媒体失败(PNG→PPM 转码)。
pub const UPLOAD_PREP_FAILED: &str = "upload_prep_failed";

// ───── queue / concurrency ────────────────────────────────────────
/// 任务槽位已满(> max_concurrent_jobs + max_queue_depth)。
pub const QUEUE_FULL: &str = "queue_full";
/// SSE 连接数超限(> 128)。
pub const SSE_LIMIT: &str = "sse_limit";
/// per-IP 限流命中(60/600 req/min)。
pub const RATE_LIMITED: &str = "rate_limited";

// ───── last-resort internal ───────────────────────────────────────
/// run_job panic 兜底。
pub const INTERNAL_PANIC: &str = "internal_panic";
/// 分配失败(oom / arena 满)。生产几乎不可能,但留位。
pub const INTERNAL_ALLOC_FAILED: &str = "internal_alloc_failed";
/// tokio runtime / tokio::spawn 失败 / join error。
pub const INTERNAL_RUNTIME: &str = "internal_runtime";

/// HTTP 状态码 → 推荐错误码的默认映射。
///
/// 让 handler 在写 `error_response_with(...)` 时只传 code,不用重复记 status:
/// - 4xx 客户端错默认 → `unknown`(待人工 review 补语义码)
/// - 5xx 服务端错默认 → `internal`
/// - 已知的 404 在调用方显式覆盖为 `job_not_found` 等。
pub fn default_code_for_status(status: axum::http::StatusCode) -> &'static str {
    use axum::http::StatusCode as S;
    match status {
        S::NOT_FOUND => UNKNOWN,
        S::BAD_REQUEST
        | S::PAYLOAD_TOO_LARGE
        | S::UNPROCESSABLE_ENTITY
        | S::RANGE_NOT_SATISFIABLE => PARAM_INVALID,
        S::UNAUTHORIZED => AUTH_MISSING,
        S::FORBIDDEN => SSRF_BLOCKED,
        S::TOO_MANY_REQUESTS => QUEUE_FULL,
        S::GONE => MEDIA_GONE,
        _ => INTERNAL,
    }
}

/// 把任意 `&str` 错误消息"归类"到最近的语义码 — 只在兜底路径里使用
/// (例如 panic message / 第三方库原始错误)。不做真正的 NLP,只是关键字匹配,
/// 避免全量错误消息解析增加热路径开销。
///
/// **顺序敏感**:具体子分类(pool-timeout)必须先于泛分类(timeout)匹配,
/// 否则 "pool timed out" 会被误判成 decode_timeout。
pub fn classify_error_message(msg: &str) -> &'static str {
    let m = msg.to_ascii_lowercase();
    // pg — 具体子分类必须先匹配。注意 PG 错误信息常用 "timed out" 而非 "timeout",
    // 因此这里把三个常见短语都纳进 pool-acquire 检测。
    if m.contains("pool") && (m.contains("timeout") || m.contains("timed out") || m.contains("acquire")) {
        return PG_TIMEOUT;
    }
    if m.contains("postgres") || m.contains("sqlx") || m.contains("pg: ") {
        return PG_UNAVAILABLE;
    }
    if m.contains("constraint") || m.contains("duplicate key") || m.contains("foreign key") {
        return PG_CONSTRAINT_VIOLATION;
    }
    // s3
    if m.contains("signature") && m.contains("does not match") {
        return S3_SIGNATURE_MISMATCH;
    }
    if m.contains("nosuchbucket") || m.contains("bucket does not exist") {
        return S3_BUCKET_MISSING;
    }
    if m.contains("nosuchkey") || m.contains("object not found") {
        return S3_OBJECT_NOT_FOUND;
    }
    if m.contains("rustfs") || m.contains("s3 ") || m.contains("transport:") {
        return S3_UNAVAILABLE;
    }
    // cascade / detector
    if m.contains("cascade") && (m.contains("not found") || m.contains("missing")) {
        return CASCADE_MISSING;
    }
    if m.contains("cascade") && (m.contains("invalid") || m.contains("corrupt")) {
        return CASCADE_CORRUPT;
    }
    if m.contains("cascade") {
        return CASCADE_LOAD_FAILED;
    }
    if m.contains("unsupported algo") {
        return UNSUPPORTED_ALGO;
    }
    // decode — truncated/corrupt 必须在 timeout 之前(因为 "image data truncated" 是常见的 decode 失败)
    if m.contains("truncated") || m.contains("early eof") || m.contains("unexpected eof") {
        return DECODE_TRUNCATED;
    }
    if m.contains("invalid data") || m.contains("corrupt") || m.contains("bad magic") {
        return DECODE_CORRUPT;
    }
    if m.contains("unsupported") || m.contains("unknown format") || m.contains("no decoder") {
        return DECODE_UNSUPPORTED;
    }
    // timeout 兜底(已经被 pg pool timeout 排除后,只剩 decode/general timeout)
    if m.contains("timeout") || m.contains("timed out") {
        return DECODE_TIMEOUT;
    }
    // queue
    if m.contains("queue") && (m.contains("full") || m.contains("depth")) {
        return QUEUE_FULL;
    }
    // panic
    if m.starts_with("panic") {
        return INTERNAL_PANIC;
    }
    INTERNAL
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_codes_are_snake_case_ascii() {
        let codes = [
            INTERNAL,
            UNKNOWN,
            CASCADE_MISSING,
            CASCADE_CORRUPT,
            CASCADE_LOAD_FAILED,
            UNSUPPORTED_ALGO,
            PG_UNAVAILABLE,
            PG_TIMEOUT,
            PG_CONSTRAINT_VIOLATION,
            S3_UNAVAILABLE,
            S3_SIGNATURE_MISMATCH,
            S3_BUCKET_MISSING,
            S3_OBJECT_NOT_FOUND,
            DECODE_UNSUPPORTED,
            DECODE_TRUNCATED,
            DECODE_CORRUPT,
            DECODE_TIMEOUT,
            JOB_NOT_FOUND,
            JOB_ALREADY_DONE,
            JOB_CANCELLED,
            JOB_PANIC,
            PARAM_OUT_OF_RANGE,
            PARAM_MISSING,
            PARAM_INVALID,
            PARAM_BATCH_TOO_LARGE,
            AUTH_MISSING,
            AUTH_INVALID,
            AUTH_EXPIRED,
            SSRF_BLOCKED,
            MEDIA_BAD_KEY,
            MEDIA_GONE,
            MEDIA_RANGE_INVALID,
            MEDIA_RANGE_UNSATISFIABLE,
            UPLOAD_TOO_LARGE,
            UPLOAD_EMPTY,
            UPLOAD_PREP_FAILED,
            QUEUE_FULL,
            SSE_LIMIT,
            RATE_LIMITED,
            INTERNAL_PANIC,
            INTERNAL_ALLOC_FAILED,
            INTERNAL_RUNTIME,
        ];
        assert!(codes.len() >= 30, "code catalog should be substantial");
        for c in codes {
            assert!(!c.is_empty());
            assert!(c.len() <= 32, "code too long: {c}");
            assert!(
                c.chars()
                    .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_'),
                "non-snake_case code: {c}"
            );
            assert!(!c.starts_with('_'), "leading underscore: {c}");
            assert!(!c.ends_with('_'), "trailing underscore: {c}");
            // 必须含至少一个字母,纯数字/下划线串不可读
            assert!(
                c.chars().any(|ch| ch.is_ascii_alphabetic()),
                "no letters: {c}"
            );
        }
        // 没有重复
        let mut sorted: Vec<&str> = codes.to_vec();
        sorted.sort();
        for w in sorted.windows(2) {
            assert_ne!(w[0], w[1], "duplicate code: {}", w[0]);
        }
    }

    #[test]
    fn default_code_for_status_covers_known_branches() {
        use axum::http::StatusCode as S;
        assert_eq!(default_code_for_status(S::NOT_FOUND), UNKNOWN);
        assert_eq!(default_code_for_status(S::BAD_REQUEST), PARAM_INVALID);
        assert_eq!(default_code_for_status(S::UNAUTHORIZED), AUTH_MISSING);
        assert_eq!(default_code_for_status(S::FORBIDDEN), SSRF_BLOCKED);
        assert_eq!(default_code_for_status(S::TOO_MANY_REQUESTS), QUEUE_FULL);
        assert_eq!(default_code_for_status(S::GONE), MEDIA_GONE);
        assert_eq!(default_code_for_status(S::INTERNAL_SERVER_ERROR), INTERNAL);
    }

    #[test]
    fn classify_pg_pool_timeout() {
        assert_eq!(
            classify_error_message("pool timed out while waiting for slot"),
            PG_TIMEOUT
        );
        assert_eq!(
            classify_error_message("acquire timeout from pool"),
            PG_TIMEOUT
        );
    }

    #[test]
    fn classify_pg_constraint_violation() {
        assert_eq!(
            classify_error_message("duplicate key value violates unique constraint"),
            PG_CONSTRAINT_VIOLATION
        );
        assert_eq!(
            classify_error_message("foreign key constraint violated"),
            PG_CONSTRAINT_VIOLATION
        );
    }

    #[test]
    fn classify_s3_signature_mismatch() {
        assert_eq!(
            classify_error_message("The request signature we calculated does not match the signature you provided"),
            S3_SIGNATURE_MISMATCH
        );
    }

    #[test]
    fn classify_s3_bucket_missing() {
        assert_eq!(
            classify_error_message("NoSuchBucket: bucket does not exist"),
            S3_BUCKET_MISSING
        );
    }

    #[test]
    fn classify_cascade_messages() {
        assert_eq!(
            classify_error_message("cascade.rfcf not found"),
            CASCADE_MISSING
        );
        assert_eq!(
            classify_error_message("cascade file corrupt: bad magic"),
            CASCADE_CORRUPT
        );
        assert_eq!(
            classify_error_message("cascade load failed (some path)"),
            CASCADE_LOAD_FAILED
        );
    }

    #[test]
    fn classify_decode_messages() {
        assert_eq!(
            classify_error_message("image data truncated"),
            DECODE_TRUNCATED
        );
        assert_eq!(
            classify_error_message("invalid data: corrupt PNG chunk"),
            DECODE_CORRUPT
        );
        assert_eq!(
            classify_error_message("ffmpeg image convert timed out after 30s"),
            DECODE_TIMEOUT
        );
        assert_eq!(
            classify_error_message("unsupported image format"),
            DECODE_UNSUPPORTED
        );
    }

    #[test]
    fn classify_panic_messages() {
        assert_eq!(
            classify_error_message("panic: index out of bounds"),
            INTERNAL_PANIC
        );
        assert_eq!(
            classify_error_message("panic at line 42: assertion failed"),
            INTERNAL_PANIC
        );
    }

    #[test]
    fn classify_falls_back_to_internal() {
        assert_eq!(classify_error_message("something totally unrelated"), INTERNAL);
        assert_eq!(classify_error_message(""), INTERNAL);
    }

    #[test]
    fn code_count_meets_target() {
        // 稳定性加固-2 的硬指标:>= 30 个语义码。
        let codes = [
            INTERNAL,
            UNKNOWN,
            CASCADE_MISSING,
            CASCADE_CORRUPT,
            CASCADE_LOAD_FAILED,
            UNSUPPORTED_ALGO,
            PG_UNAVAILABLE,
            PG_TIMEOUT,
            PG_CONSTRAINT_VIOLATION,
            S3_UNAVAILABLE,
            S3_SIGNATURE_MISMATCH,
            S3_BUCKET_MISSING,
            S3_OBJECT_NOT_FOUND,
            DECODE_UNSUPPORTED,
            DECODE_TRUNCATED,
            DECODE_CORRUPT,
            DECODE_TIMEOUT,
            JOB_NOT_FOUND,
            JOB_ALREADY_DONE,
            JOB_CANCELLED,
            JOB_PANIC,
            PARAM_OUT_OF_RANGE,
            PARAM_MISSING,
            PARAM_INVALID,
            PARAM_BATCH_TOO_LARGE,
            AUTH_MISSING,
            AUTH_INVALID,
            AUTH_EXPIRED,
            SSRF_BLOCKED,
            MEDIA_BAD_KEY,
            MEDIA_GONE,
            MEDIA_RANGE_INVALID,
            MEDIA_RANGE_UNSATISFIABLE,
            UPLOAD_TOO_LARGE,
            UPLOAD_EMPTY,
            UPLOAD_PREP_FAILED,
            QUEUE_FULL,
            SSE_LIMIT,
            RATE_LIMITED,
            INTERNAL_PANIC,
            INTERNAL_ALLOC_FAILED,
            INTERNAL_RUNTIME,
        ];
        assert!(
            codes.len() >= 35,
            "stability-hardening-2 wants >= 35 codes; got {}",
            codes.len()
        );
    }
}
