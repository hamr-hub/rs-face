/* rs-face Platform · 用户行为埋点(轻量,无依赖)
 *
 * 设计原则:
 * - 自动捕获 5 类事件:page_view / click / api_call / js_error / kpi_poll
 * - sendBeacon 兜底 visibilitychange + beforeunload,避免丢点
 * - 批量 flush:每 5s 或攒够 20 条发一次;页面隐藏 / 卸载立即 flush
 * - PII 过滤:服务端再过滤一次,客户端先扫一遍减少 waste
 * - 会话 id:sessionStorage 持久化,关 tab 换号
 *
 * 自定义事件:window.__track(name, props?) — 业务代码主动调用
 *   e.g. __track('upload_started', { kind: 'image', size_kb: 320 })
 *
 * 与 web/app.js 共享全局 state(避免重复元数据采集)。
 */
'use strict';

const SESSION_KEY = 'rsface.telemetry.session';
const MUTED_KEY = 'rsface.telemetry.muted'; // 用户关闭埋点的偏好
const BATCH_MAX = 20;
const FLUSH_INTERVAL_MS = 5000;

const FORBIDDEN = ['s3://', 'local://', 'inline://', '/media/', 'Authorization', 'Bearer '];

const telemetry = (() => {
  let session = '';
  let buffer = [];
  let flushTimer = null;
  let lastFlushAt = 0;
  let installed = false;
  let originalFetch = null; // wrap fetch 后打点

  function uuid() {
    // 简单 v4-like UUID,不用 crypto.randomUUID 是为兼容老浏览器。
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = (Math.random() * 16) | 0;
      const v = c === 'x' ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  function getSession() {
    try {
      let s = sessionStorage.getItem(SESSION_KEY);
      if (!s) { s = uuid(); sessionStorage.setItem(SESSION_KEY, s); }
      return s;
    } catch { return 'in-memory-' + Math.random().toString(36).slice(2, 12); }
  }

  function isMuted() {
    try { return localStorage.getItem(MUTED_KEY) === '1'; }
    catch { return false; }
  }

  /** 服务端兜底的安全过滤:客户端再扫一遍,减少发送的数据量。 */
  function isSafeProps(props) {
    const blob = JSON.stringify(props || {});
    return !FORBIDDEN.some(s => blob.includes(s));
  }

  /** 公共入参:任何调用方都可以 push 一条事件;内部做缓冲。 */
  function track(name, props) {
    if (isMuted()) return;
    if (!name || typeof name !== 'string' || name.length > 64) return;
    if (!isSafeProps(props)) return;
    buffer.push({
      name,
      ts: Date.now(),
      session,
      path: location.pathname,
      props: props || {},
    });
    if (buffer.length >= BATCH_MAX) flush();
  }

  /** 主动调用 sendBeacon / fetch 把 buffer 发出去。 */
  function flush() {
    if (buffer.length === 0) return;
    if (Date.now() - lastFlushAt < 50) return; // 防抖
    lastFlushAt = Date.now();
    const batch = buffer;
    buffer = [];
    const body = JSON.stringify({ events: batch });
    // sendBeacon 优先(不阻塞页面切换,浏览器保证送达)。
    if (navigator.sendBeacon) {
      try {
        const ok = navigator.sendBeacon('/api/telemetry', new Blob([body], { type: 'application/json' }));
        if (ok) return;
      } catch {}
    }
    // 退化:fetch keepalive。
    fetch('/api/telemetry', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      keepalive: true,
    }).catch(() => {});
  }

  /** 自动采集:装好 fetch wrap + 全局错误监听 + KPI 计数。 */
  function installAutoCapture() {
    if (installed) return;
    installed = true;
    session = getSession();

    // 1) fetch 拦截:抓 endpoint + status + 时长,不发 body / headers(避免敏感数据)。
    originalFetch = window.fetch.bind(window);
    window.fetch = function patchedFetch(input, init) {
      const url = typeof input === 'string' ? input : (input && input.url) || '';
      const method = (init && init.method) || (input && input.method) || 'GET';
      const t0 = Date.now();
      // 只埋点本平台 API,过滤掉 /media/ 这种大流量。
      if (url.indexOf('/api/') === 0) {
        return originalFetch(input, init).then(
          (resp) => {
            track('api_call', {
              url: stripQuery(url),
              method,
              status: resp.status,
              duration_ms: Date.now() - t0,
            });
            return resp;
          },
          (err) => {
            track('api_error', {
              url: stripQuery(url),
              method,
              duration_ms: Date.now() - t0,
              message: String(err && err.message || err).slice(0, 128),
            });
            throw err;
          }
        );
      }
      return originalFetch(input, init);
    };

    // 2) 全局 JS 错误 → 'error' 事件。
    window.addEventListener('error', (ev) => {
      track('js_error', {
        message: (ev.message || '').slice(0, 256),
        source: (ev.filename || '').slice(-128),
        line: ev.lineno, col: ev.colno,
      });
    });
    window.addEventListener('unhandledrejection', (ev) => {
      const r = ev.reason;
      track('js_unhandled', {
        message: (r && r.message ? String(r.message) : String(r)).slice(0, 256),
      });
    });

    // 3) 初始 page_view。
    track('page_view', { ref: document.referrer.slice(0, 128) || null });

    // 4) 定时 flush。
    flushTimer = setInterval(flush, FLUSH_INTERVAL_MS);

    // 5) 页面隐藏 / 关闭时 flush。
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') flush();
    });
    window.addEventListener('pagehide', flush);
    window.addEventListener('beforeunload', flush);
  }

  function stripQuery(url) {
    const i = url.indexOf('?');
    return i >= 0 ? url.slice(0, i) : url;
  }

  /** 用户从 UI 关闭/开启埋点(可选,目前未在 UI 暴露)。 */
  function setMuted(on) {
    try { localStorage.setItem(MUTED_KEY, on ? '1' : '0'); } catch {}
  }

  /** 测试 / 调试时强制 flush。 */
  function flushNow() { flush(); }

  return { installAutoCapture, track, flushNow, setMuted, get isMuted() { return isMuted(); } };
})();

/** 业务代码统一入口:window.__track(name, props) */
window.__track = (name, props) => telemetry.track(name, props);

// DOMContentLoaded 后再装,让其它模块能先初始化。
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => telemetry.installAutoCapture(), { once: true });
} else {
  telemetry.installAutoCapture();
}