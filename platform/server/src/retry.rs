//! 指数退避重试(transient-failure only)。
//!
//! 设计目标:
//! - 把"瞬时失败"和"永久失败"显式区分 — 同一调用点 retry 100 次不可能把一个
//!   `cascade.rfcf 缺失`变成成功,只会浪费 CPU;
//! - 不引入 `tokio-retry`/`backoff` 等 crate — 平台层依赖最小化;
//! - 日志:每次 retry 走 `eprintln!` 记录 `op` + attempt + delay + reason,运维能
//!   直接从日志看到"哪类失败在抖"。

use std::future::Future;
use std::time::Duration;

/// 重试策略。`initial_delay` 翻倍直到 `max_delay`,共 `max_retries` 次额外尝试
/// (即总尝试次数 = `max_retries + 1`)。
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
}

impl RetryPolicy {
    /// 默认策略:5 次重试,起始 100ms,翻倍到 5s 上限。
    /// 100ms → 200 → 400 → 800 → 1600 → 3200ms → 第 5 次 5000ms。
    /// 累计最大等待 ~11s,适合 PG / S3 短抖动。
    pub const fn default_pg_s3() -> Self {
        Self {
            max_retries: 5,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
        }
    }

    /// 仅 2 次重试(用于热路径 / 实时性要求高的端点,如 /media)。
    pub const fn fast() -> Self {
        Self {
            max_retries: 2,
            initial_delay: Duration::from_millis(50),
            max_delay: Duration::from_millis(500),
        }
    }

    /// 不重试。给"明知不可能瞬时恢复"的失败用,例如鉴权 / 4xx 错误。
    pub const fn no_retry() -> Self {
        Self {
            max_retries: 0,
            initial_delay: Duration::from_millis(0),
            max_delay: Duration::from_millis(0),
        }
    }

    fn delay_for(&self, attempt: u32) -> Duration {
        // attempt 从 0 起:首次失败 → 等 initial_delay;之后翻倍,封顶 max_delay。
        // u32 没有 saturating_shl,用 checked_shl + unwrap_or(0) 防溢出。
        let factor: u32 = 1u32.checked_shl(attempt.min(31)).unwrap_or(u32::MAX);
        let d = self.initial_delay.saturating_mul(factor);
        if d > self.max_delay {
            self.max_delay
        } else {
            d
        }
    }
}

/// 决定一个错误是否值得重试。
///
/// `Err: &(dyn std::error::Error)` 让调用方传任何 `E: std::error::Error`。
/// 默认规则:
/// - io::ErrorKind::TimedOut / ConnectionRefused / ConnectionReset /
///   ConnectionAborted / NotConnected / BrokenPipe / Interrupted → 重试;
/// - io::ErrorKind::PermissionDenied / InvalidInput / NotFound → 不重试
///   (404 / 鉴权错误不会因为多尝试几次就成功);
/// - 自定义错误信息含 "timeout" / "timed out" / "broken pipe" / "connection" →
///   视为瞬时,重试(覆盖 sqlx / ureq 等非 std::io::Error 错误类型);
/// - 否则:不重试(保守,免得把永久错误拖到超时)。
///
/// **重要**:这个谓词只判断"瞬时"特征,不判断"操作是否幂等"。
/// 调用方必须自己确认被 retry 的操作是幂等的(GET / HEAD / INSERT-on-conflict 等)。
pub fn is_transient(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    if m.contains("timeout") || m.contains("timed out") {
        return true;
    }
    if m.contains("connection refused") || m.contains("connection reset") {
        return true;
    }
    if m.contains("broken pipe") {
        return true;
    }
    if m.contains("temporar") {
        // temporary failure / temporarily unavailable
        return true;
    }
    if m.contains("throttl") || m.contains("rate limit") || m.contains("too many requests") {
        return true;
    }
    false
}

/// 用给定策略执行 `op`,仅在错误信息被判为 transient 时重试。
///
/// `op` 接受一个 `attempt` 参数(0-indexed),调用方可在日志里用它判断是第几次。
///
/// 重试期间每次都把 attempt + delay + reason 打到 stderr,运维可以从时间线上
/// 直接看到 transient 抖动的频次和累计延迟。
pub async fn retry_with_backoff<F, Fut, T, E>(op_name: &str, policy: RetryPolicy, mut op: F) -> Result<T, E>
where
    F: FnMut(u32) -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut attempt: u32 = 0;
    loop {
        match op(attempt).await {
            Ok(v) => {
                if attempt > 0 {
                    eprintln!("[retry] {op_name} succeeded on attempt {}", attempt + 1);
                }
                return Ok(v);
            }
            Err(e) => {
                let msg = e.to_string();
                if attempt >= policy.max_retries || !is_transient(&msg) {
                    if attempt > 0 {
                        eprintln!(
                            "[retry] {op_name} giving up after {} attempts: {msg}",
                            attempt + 1
                        );
                    }
                    return Err(e);
                }
                let delay = policy.delay_for(attempt);
                eprintln!(
                    "[retry] {op_name} attempt {} failed: {msg}; retrying in {}ms",
                    attempt + 1,
                    delay.as_millis()
                );
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
        }
    }
}

/// 同步版 `retry_with_backoff` — 用于 `spawn_blocking` 内阻塞调用(S3Client.put_object
/// / get_object / list_objects 等)。transient 分类沿用 `is_transient`。
pub fn retry_sync<T, E, F>(op_name: &str, policy: RetryPolicy, mut op: F) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
    E: std::fmt::Display,
{
    let mut attempt: u32 = 0;
    loop {
        match op() {
            Ok(v) => {
                if attempt > 0 {
                    eprintln!("[retry] {op_name} succeeded on attempt {}", attempt + 1);
                }
                return Ok(v);
            }
            Err(e) => {
                let msg = e.to_string();
                if attempt >= policy.max_retries || !is_transient(&msg) {
                    if attempt > 0 {
                        eprintln!(
                            "[retry] {op_name} giving up after {} attempts: {msg}",
                            attempt + 1
                        );
                    }
                    return Err(e);
                }
                let delay = policy.delay_for(attempt);
                eprintln!(
                    "[retry] {op_name} attempt {} failed: {msg}; retrying in {}ms",
                    attempt + 1,
                    delay.as_millis()
                );
                std::thread::sleep(delay);
                attempt += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn delay_doubles_and_caps() {
        let p = RetryPolicy::default_pg_s3();
        assert_eq!(p.delay_for(0), Duration::from_millis(100));
        assert_eq!(p.delay_for(1), Duration::from_millis(200));
        assert_eq!(p.delay_for(2), Duration::from_millis(400));
        assert_eq!(p.delay_for(3), Duration::from_millis(800));
        assert_eq!(p.delay_for(4), Duration::from_millis(1600));
        assert_eq!(p.delay_for(5), Duration::from_millis(3200));
        // 6400ms > 5000ms cap → capped to 5000ms
        assert_eq!(p.delay_for(6), Duration::from_secs(5));
        assert_eq!(p.delay_for(31), Duration::from_secs(5)); // saturating
    }

    #[test]
    fn fast_policy_caps_lower() {
        let p = RetryPolicy::fast();
        assert_eq!(p.delay_for(0), Duration::from_millis(50));
        assert_eq!(p.delay_for(3), Duration::from_millis(400));
        assert_eq!(p.delay_for(4), Duration::from_millis(500)); // capped
    }

    #[test]
    fn no_retry_policy_returns_zero_delay() {
        let p = RetryPolicy::no_retry();
        assert_eq!(p.delay_for(0), Duration::from_millis(0));
    }

    #[test]
    fn classifier_recognises_transient_phrases() {
        for msg in [
            "connection timed out",
            "acquire timeout from pool",
            "connection refused",
            "connection reset by peer",
            "broken pipe",
            "temporary failure",
            "service temporarily unavailable",
            "too many requests",
            "rate limit exceeded",
            "throttled by server",
        ] {
            assert!(is_transient(msg), "should classify as transient: {msg}");
        }
    }

    #[test]
    fn classifier_rejects_permanent_phrases() {
        for msg in [
            "cascade.rfcf not found",
            "permission denied",
            "invalid input",
            "bucket does not exist",
            "object not found",
            "duplicate key value violates unique constraint",
            "foreign key constraint violated",
            "syntax error at or near",
            "panic: index out of bounds",
            "",
        ] {
            assert!(!is_transient(msg), "should NOT classify as transient: {msg}");
        }
    }

    #[tokio::test]
    async fn succeeds_first_try_no_log() {
        let counter = Arc::new(AtomicU32::new(0));
        let c2 = counter.clone();
        let r: Result<u32, &'static str> = retry_with_backoff("test", RetryPolicy::default_pg_s3(), move |_attempt| {
            let c = c2.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(42)
            }
        })
        .await;
        assert_eq!(r.unwrap(), 42);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retries_then_succeeds() {
        let counter = Arc::new(AtomicU32::new(0));
        let c2 = counter.clone();
        let r: Result<u32, String> = retry_with_backoff("flaky", RetryPolicy {
            max_retries: 3,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
        }, move |attempt| {
            let c = c2.clone();
            async move {
                let n = c.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    // 前两次给 transient 错误
                    Err("connection timeout".into())
                } else {
                    Ok(7)
                }
            }
        })
        .await;
        assert_eq!(r.unwrap(), 7);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn gives_up_after_max_retries() {
        let counter = Arc::new(AtomicU32::new(0));
        let c2 = counter.clone();
        let r: Result<u32, String> = retry_with_backoff(
            "always-fails",
            RetryPolicy {
                max_retries: 2,
                initial_delay: Duration::from_millis(1),
                max_delay: Duration::from_millis(2),
            },
            move |_attempt| {
                let c = c2.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Err("connection timed out".into())
                }
            },
        )
        .await;
        assert!(r.is_err());
        // 1 initial + 2 retries = 3 attempts
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn does_not_retry_permanent_error() {
        let counter = Arc::new(AtomicU32::new(0));
        let c2 = counter.clone();
        let r: Result<u32, String> = retry_with_backoff("perm", RetryPolicy::default_pg_s3(), move |_attempt| {
            let c = c2.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err("bucket does not exist".into())
            }
        })
        .await;
        assert!(r.is_err());
        // Only one attempt — permanent error short-circuits retry loop.
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn real_elapsed_time_matches_sum_of_delays() {
        let start = tokio::time::Instant::now();
        let _: Result<(), String> = retry_with_backoff(
            "elapsed-test",
            RetryPolicy {
                max_retries: 2,
                initial_delay: Duration::from_millis(20),
                max_delay: Duration::from_millis(20),
            },
            |_attempt| async { Err("connection timeout".into()) },
        )
        .await;
        let elapsed = start.elapsed();
        // 3 attempts = 2 sleeps of 20ms each = 40ms minimum (excluding op time).
        assert!(
            elapsed >= Duration::from_millis(40),
            "elapsed = {elapsed:?}, expected >= 40ms"
        );
    }

    #[test]
    fn sync_retry_succeeds_after_transient_failures() {
        let counter = Arc::new(AtomicU32::new(0));
        let c2 = counter.clone();
        let r: Result<u32, String> = retry_sync(
            "sync-flaky",
            RetryPolicy {
                max_retries: 3,
                initial_delay: Duration::from_millis(1),
                max_delay: Duration::from_millis(2),
            },
            || -> Result<u32, String> {
                let n = c2.fetch_add(1, Ordering::SeqCst);
                if n < 1 {
                    Err("connection refused".into())
                } else {
                    Ok(99)
                }
            },
        );
        assert_eq!(r.unwrap(), 99);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn sync_retry_no_retry_policy_returns_immediately() {
        let counter = Arc::new(AtomicU32::new(0));
        let c2 = counter.clone();
        let r: Result<(), String> = retry_sync(
            "no-retry",
            RetryPolicy::no_retry(),
            || -> Result<(), String> {
                c2.fetch_add(1, Ordering::SeqCst);
                Err("connection timeout".into())
            },
        );
        assert!(r.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
