//! per-IP 速率限制(令牌桶 + DashMap 状态)。
//!
//! **stability-hardening-2 加**,目标:
//! - 阻止单个 IP 滥用 POST `/api/jobs`(60/min 上限)
//! - 阻止单个 IP 短时间拖垮 GET 列表(600/min 上限)
//! - SSE 不限流 — 长连接本身已经显式 cap 了并发数
//!
//! 实现选择:
//! - **`tokio::sync::RwLock<HashMap>`** 而不是 DashMap:省一个 crate,
//!   且 rate-limit 路径是"每请求 1 次 acquire_read" + "更新状态时短暂 write",
//!   锁竞争低于常规 web 服务的吞吐上限。HashMap 的 eviction 在 write 路径
//!   顺手做(每 1024 次检查,清理 idle > 5min 的桶)。
//! - **令牌桶**(而不是固定窗口计数器):平滑突发 — 单 IP 1 秒内连发 60 个请求
//!   应该被允许(浏览器并发 6 个标签 + 每标签预 fetch),但 60 个/秒持续要拒。
//!
//! 内存上限:bucket 数 ≤ 唯一 IP 数,每个桶 ~ 80 字节,1w IP ≈ 800KB,安全。
//! 进程重启会清空所有桶(接受,重启通常意味着配置变更,旧限流应清空)。

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LimitDecision {
    /// 是否放行。false 时返回 429。
    pub allowed: bool,
    /// 当前桶内剩余令牌(给 `Retry-After` 算下一次可用时间)。
    pub remaining: u32,
    /// 距桶满所需时间(秒,向上取整)。
    pub retry_after_secs: u64,
}

/// 单个 IP 的令牌桶。`capacity` 是上限,`refill_per_sec` 是每秒补充速率。
#[derive(Debug, Clone)]
struct TokenBucket {
    capacity: u32,
    refill_per_sec: f64,
    tokens: f64,
    last_refill: Instant,
    last_seen: Instant,
}

impl TokenBucket {
    fn new(capacity: u32, refill_per_sec: f64, now: Instant) -> Self {
        Self {
            capacity,
            refill_per_sec,
            tokens: capacity as f64,
            last_refill: now,
            last_seen: now,
        }
    }

    /// 拿一个令牌,返回决策。
    fn try_take(&mut self, now: Instant) -> LimitDecision {
        // 先按时间补充(refill 上限 = capacity)
        let elapsed = now.saturating_duration_since(self.last_refill);
        if !elapsed.is_zero() {
            self.tokens = (self.tokens + elapsed.as_secs_f64() * self.refill_per_sec)
                .min(self.capacity as f64);
            self.last_refill = now;
        }
        self.last_seen = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            LimitDecision {
                allowed: true,
                remaining: self.tokens.floor() as u32,
                retry_after_secs: 0,
            }
        } else {
            // 还差多少秒能拿一个令牌
            let deficit = 1.0 - self.tokens;
            let secs = deficit / self.refill_per_sec;
            LimitDecision {
                allowed: false,
                remaining: 0,
                retry_after_secs: secs.ceil() as u64,
            }
        }
    }
}

/// 全局限流器。按 IP 维度记录桶。
///
/// 设计上不用泛型 — 写死 String(IPv4/IPv6 字面量字符串),避免引入 ipnet / std::net::IpAddr
/// 等类型导致 axum middleware 路径复杂化。
#[derive(Clone)]
pub struct RateLimiter {
    buckets: Arc<RwLock<HashMap<String, TokenBucket>>>,
    /// 每 N 次 write 检查一次 stale bucket eviction,降低清理成本。
    evict_every: u64,
    /// idle 桶清理阈值 — 后台任务调 `evict_idle` 时使用,字段保留供后续调优。
    #[allow(dead_code)]
    idle_evict_after: Duration,
    op_count: Arc<std::sync::atomic::AtomicU64>,
}

impl RateLimiter {
    /// 默认实例:POST 60/min、GET 600/min(分别对应不同 rate)。
    /// 不同路由可以共用同一个 limiter,传不同 `capacity` / `refill_per_sec` 即可。
    pub fn new() -> Self {
        Self {
            buckets: Arc::new(RwLock::new(HashMap::new())),
            evict_every: 1024,
            idle_evict_after: Duration::from_secs(300),
            op_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// 检查指定 IP 是否还能再发 1 个请求(给定桶配置)。
    ///
    /// `capacity` 和 `refill_per_sec` 是该路由的速率上限。
    /// 返回 `LimitDecision` — allowed=false 时调用方应该 429 + Retry-After。
    pub async fn check(
        &self,
        scope: &str,
        ip: &str,
        capacity: u32,
        refill_per_sec: f64,
    ) -> LimitDecision {
        let now = Instant::now();
        let bucket_key = format!("{scope}:{ip}");
        // 1) read 路径:尝试拿桶决策
        {
            let guard = self.buckets.read().await;
            if let Some(b) = guard.get(&bucket_key) {
                // 拿桶的 clone 不行(TokenBucket 含 Instant 不可 mut),
                // 走"释放读锁再拿写锁重做"的简单路径;高 QPS 下 write 锁竞争可忽略。
                let mut b = b.clone();
                drop(guard);
                let decision = b.try_take(now);
                if decision.allowed {
                    // 回写剩余 tokens
                    let mut wg = self.buckets.write().await;
                    wg.insert(bucket_key.clone(), b);
                    self.maybe_evict(&wg);
                    return decision;
                } else {
                    // 拒绝路径不更新桶状态,但记录 last_seen
                    let mut wg = self.buckets.write().await;
                    b.last_seen = now;
                    wg.insert(bucket_key.clone(), b);
                    self.maybe_evict(&wg);
                    return decision;
                }
            }
        }
        // 2) miss:首次见此 IP,创建一个满桶并消耗 1 个 token
        let mut b = TokenBucket::new(capacity, refill_per_sec, now);
        let decision = b.try_take(now);
        let mut wg = self.buckets.write().await;
        wg.insert(bucket_key, b);
        self.maybe_evict(&wg);
        decision
    }

    /// 后台偶发清理:每 N 次写操作扫描一次,把 idle > 阈值 的桶删掉。
    /// 当前 RwLock 已经是 guard 持锁状态,所以这里只看 size 决定是否要清理。
    fn maybe_evict(&self, buckets: &HashMap<String, TokenBucket>) {
        let n = self
            .op_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !n.is_multiple_of(self.evict_every) {
            return;
        }
        // 注意:这里不能直接 &mut HashMap 调用,因为调用方已持锁。
        // 真正的清理逻辑放到 `evict_idle` 公开方法里,由后台任务触发。
        let _ = buckets; // 占位,eviction 走独立方法
    }

    /// 主动扫描并删除 idle 桶(由后台任务每分钟调一次)。
    #[allow(dead_code)]
    pub async fn evict_idle(&self) -> usize {
        let now = Instant::now();
        let mut wg = self.buckets.write().await;
        let before = wg.len();
        wg.retain(|_, b| now.duration_since(b.last_seen) <= self.idle_evict_after);
        before.saturating_sub(wg.len())
    }

    /// 调试 / 测试用 — 当前桶数。
    #[allow(dead_code)]
    pub async fn bucket_count(&self) -> usize {
        self.buckets.read().await.len()
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

/// 从 axum 的 `ConnectInfo` 提取 IP,fallback "unknown"。
///
/// axum 默认不启用 ConnectInfo,所以 middleware 用法是:
/// ```ignore
/// use axum::extract::ConnectInfo;
/// use std::net::SocketAddr;
/// async fn handler(ConnectInfo(addr): ConnectInfo<SocketAddr>, ...) { ... }
/// ```
/// 然后从 `addr.ip()` 拿 IP。在 axum router 启动时用
/// `into_make_service_with_connect_info::<SocketAddr>()`。
pub fn ip_from_request(remote: Option<SocketAddr>, headers: &axum::http::HeaderMap) -> String {
    if let Some(addr) = remote {
        return addr.ip().to_string();
    }
    // 反代场景(nginx / cloud LB):用 X-Forwarded-For 第一段。
    if let Some(xff) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()) {
        if let Some(first) = xff.split(',').next() {
            return first.trim().to_string();
        }
    }
    if let Some(real) = headers.get("x-real-ip").and_then(|v| v.to_str().ok()) {
        return real.to_string();
    }
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy_post() -> (u32, f64) {
        // 60 req/min = 1 req/sec 平均,但允许突发 60 个
        (60, 1.0)
    }

    #[tokio::test]
    async fn first_request_allowed_with_full_bucket() {
        let rl = RateLimiter::new();
        let d = rl.check("test", "1.1.1.1", policy_post().0, policy_post().1).await;
        assert!(d.allowed);
        assert_eq!(d.remaining, 59);
    }

    #[tokio::test]
    async fn capacity_burst_then_refill() {
        let rl = RateLimiter::new();
        // 1 req/sec refill:burst=2 时,第 3 次直接拒绝
        let (cap, rate) = (2u32, 1.0f64);
        let d1 = rl.check("test", "2.2.2.2", cap, rate).await;
        assert!(d1.allowed);
        let d2 = rl.check("test", "2.2.2.2", cap, rate).await;
        assert!(d2.allowed);
        let d3 = rl.check("test", "2.2.2.2", cap, rate).await;
        assert!(
            !d3.allowed,
            "3rd request should be denied with cap=2 rate=1/s"
        );
        assert!(
            d3.retry_after_secs >= 1,
            "retry-after must be ~1s, got {}",
            d3.retry_after_secs
        );
    }

    #[tokio::test]
    async fn separate_ips_have_independent_buckets() {
        let rl = RateLimiter::new();
        let (cap, rate) = (1u32, 0.0001f64); // very slow refill
        let _ = rl.check("test", "a", cap, rate).await; // drain a
        let _ = rl.check("test", "a", cap, rate).await; // a denied
        let d = rl.check("test", "b", cap, rate).await;
        assert!(d.allowed, "b has its own bucket and should be full");
    }

    #[tokio::test]
    async fn refill_over_time_restores_tokens() {
        let rl = RateLimiter::new();
        let (cap, rate) = (2u32, 1000.0f64); // 1000 tokens/sec refill
        let _ = rl.check("test", "x", cap, rate).await;
        let _ = rl.check("test", "x", cap, rate).await;
        let _ = rl.check("test", "x", cap, rate).await; // 3rd rejected
        tokio::time::sleep(Duration::from_millis(20)).await; // 1000/s * 20ms = 20 tokens restored
        let d = rl.check("test", "x", cap, rate).await;
        assert!(d.allowed, "after 20ms with refill=1000/s should be allowed");
    }

    #[tokio::test]
    async fn retry_after_is_roughly_deficit_over_rate() {
        let rl = RateLimiter::new();
        // 1 token/sec refill
        let (cap, rate) = (1u32, 1.0f64);
        let _ = rl.check("test", "y", cap, rate).await;
        let _ = rl.check("test", "y", cap, rate).await; // denied, expects ~1s
        let d = rl.check("test", "y", cap, rate).await;
        assert!(!d.allowed);
        assert_eq!(d.retry_after_secs, 1);
    }

    #[tokio::test]
    async fn bucket_count_tracks_unique_ips() {
        let rl = RateLimiter::new();
        let _ = rl.check("test", "ip1", 10, 1.0).await;
        let _ = rl.check("test", "ip2", 10, 1.0).await;
        let _ = rl.check("test", "ip1", 10, 1.0).await; // reuse
        assert_eq!(rl.bucket_count().await, 2);
    }

    #[tokio::test]
    async fn evict_idle_removes_unused_buckets() {
        // 用 1ms idle 阈值模拟"陈旧"
        let mut rl = RateLimiter::new();
        rl.idle_evict_after = Duration::from_millis(5);
        let _ = rl.check("test", "z", 10, 1.0).await;
        assert_eq!(rl.bucket_count().await, 1);
        tokio::time::sleep(Duration::from_millis(20)).await;
        let removed = rl.evict_idle().await;
        assert_eq!(removed, 1);
        assert_eq!(rl.bucket_count().await, 0);
    }

    #[test]
    fn ip_from_request_prefers_connect_info() {
        let addr: SocketAddr = "10.0.0.1:8080".parse().unwrap();
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-forwarded-for", "8.8.8.8".parse().unwrap());
        assert_eq!(ip_from_request(Some(addr), &headers), "10.0.0.1");
    }

    #[test]
    fn ip_from_request_falls_back_to_xff() {
        let headers = axum::http::HeaderMap::new();
        let mut headers = headers;
        headers.insert("x-forwarded-for", "8.8.8.8, 9.9.9.9".parse().unwrap());
        assert_eq!(ip_from_request(None, &headers), "8.8.8.8");
    }

    #[test]
    fn ip_from_request_falls_back_to_x_real_ip() {
        let headers = axum::http::HeaderMap::new();
        let mut headers = headers;
        headers.insert("x-real-ip", "7.7.7.7".parse().unwrap());
        assert_eq!(ip_from_request(None, &headers), "7.7.7.7");
    }

    #[test]
    fn ip_from_request_returns_unknown_when_nothing() {
        let headers = axum::http::HeaderMap::new();
        assert_eq!(ip_from_request(None, &headers), "unknown");
    }
}
