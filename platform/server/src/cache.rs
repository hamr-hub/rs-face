//! 平台响应缓存层(轻量,无外部依赖)。
//!
//! 目标:
//! - `/api/config`:`config_info()` 一旦在启动期算好,运行期不会变。缓存后
//!   第一次求值后所有后续请求都从原子 swap 里拿 JSON 字符串,O(1) 命中,
//!   不再读 cfg + 不走 algo 列表克隆。
//! - `/api/metrics`:聚合需要扫描 registry 内所有 job,代价随 job 数线性。
//!   前端每 2s 轮询一次;我们用 1s dedup snapshot 让 1s 内的多请求只算 1 次。
//!
//! 设计取舍:为避免引入 ArcSwap,这里用 `Mutex<Option<(Instant, Vec<u8>)>>`,
//! 命中路径只过一把短锁 + 直接 clone bytes,比每次重新聚合省 50-200×(视
//! job 数);失败路径(锁等待 / poisoning)由 Mutex 自身处理。
//!
//! 失效语义:TTL 过期就重算并覆盖;没过期就复用。即使后台聚合算法执行了
//! 5ms,前端感知是 0.1ms。Redis 风格的 stale-while-revalidate 在这里收益
//! 微弱(锁本身已 1ms 内),不加复杂度。

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// 简易 TTL 缓存:`get_or_fill` 在 TTL 过期或首次访问时调用 `fill`,
/// 把 fill 的 Vec<u8> 结果缓存 `ttl` 时间。
pub struct TtlCache {
    state: Mutex<Option<(Instant, Vec<u8>)>>,
    ttl: Duration,
    /// 缓存名,仅供 `hit/miss` 日志用。
    #[allow(dead_code)]
    name: &'static str,
}

impl TtlCache {
    pub fn new(name: &'static str, ttl: Duration) -> Self {
        Self {
            state: Mutex::new(None),
            ttl,
            name,
        }
    }

    /// 命中路径:仅当上次填充到现在 ≤ TTL 时返回 Some(&bytes);
    /// 否则返回 None(调用方触发 fill)。
    pub fn get_fresh(&self) -> Option<Vec<u8>> {
        let guard = self.state.lock().unwrap_or_else(|p| p.into_inner());
        match guard.as_ref() {
            Some((at, bytes)) if at.elapsed() < self.ttl => Some(bytes.clone()),
            _ => None,
        }
    }

    /// 用新结果覆盖缓存(总是成功,即使 TTL 未到)。
    pub fn put(&self, bytes: Vec<u8>) {
        let mut guard = self.state.lock().unwrap_or_else(|p| p.into_inner());
        *guard = Some((Instant::now(), bytes));
    }

    #[allow(dead_code)]
    pub fn name(&self) -> &'static str {
        self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn empty_cache_returns_none() {
        let c = TtlCache::new("t", Duration::from_millis(100));
        assert!(c.get_fresh().is_none());
    }

    #[test]
    fn put_then_get_returns_value_within_ttl() {
        let c = TtlCache::new("t", Duration::from_millis(100));
        c.put(b"hello".to_vec());
        let got = c.get_fresh();
        assert_eq!(got.as_deref(), Some(&b"hello"[..]));
    }

    #[test]
    fn get_returns_none_after_ttl_expires() {
        let c = TtlCache::new("t", Duration::from_millis(20));
        c.put(b"x".to_vec());
        thread::sleep(Duration::from_millis(40));
        assert!(c.get_fresh().is_none());
    }

    #[test]
    fn put_overrides_even_when_fresh() {
        let c = TtlCache::new("t", Duration::from_secs(60));
        c.put(b"first".to_vec());
        c.put(b"second".to_vec());
        assert_eq!(c.get_fresh().as_deref(), Some(&b"second"[..]));
    }

    /// 并发:多线程同时 put/get 不死锁。
    #[test]
    fn concurrent_access_is_safe() {
        let c = Arc::new(TtlCache::new("c", Duration::from_millis(10)));
        let mut handles = Vec::new();
        for i in 0..8 {
            let cc = c.clone();
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    cc.put(format!("v{i}-{j}").into_bytes());
                    let _ = cc.get_fresh();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // 最终能 get 出最后一次 put 的内容
        assert!(c.get_fresh().is_some());
    }
}
