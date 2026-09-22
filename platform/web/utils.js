// 通用工具集 — 必须在 app.js 之前加载(toast/modal/dashboard/upload-queue/
// dropzone-preview 等文件都依赖 `utils`,而它们又用 defer 在 app.js 之前执行)。
const utils = (() => {
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  function legacyToast(msg, isError) {
    const el = $('#toast'); if (!el) return;
    el.textContent = msg; el.classList.toggle('error', !!isError); el.classList.remove('hidden');
    clearTimeout(el._t); el._t = setTimeout(() => el.classList.add('hidden'), 3200);
  }
  function fmtTime(ms) {
    if (ms == null) return '--:--';
    const s = ms / 1000, m = Math.floor(s / 60), sec = Math.floor(s % 60), frac = Math.floor((s % 1) * 10);
    return `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}.${frac}`;
  }
  const fmtAbsTime = ms => ms ? new Date(ms).toLocaleString() : '';
  const escapeHtml = s => String(s).replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
  /** 转义后,把所有命中 query 的字符段包成 <mark>。query 为空时原样返回。 */
  function highlight(text, query) {
    const safe = escapeHtml(text || '');
    if (!query) return safe;
    let re;
    try { re = new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'); }
    catch { return safe; }
    return safe.replace(re, m => '<mark>' + m + '</mark>');
  }
  function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
  function throttleRaf(fn) { let s = false, la = null; return (...a) => { la = a; if (s) return; s = true; requestAnimationFrame(() => { s = false; fn(...la); }); }; }
  /** Bug 1/4: URL-encode media keys so 'local://jobs/...' works in <video>/<img>. */
  function mediaUrl(key) {
    if (!key) return '';
    if (/^(https?:|data:|blob:)/.test(key)) return key;
    return '/media/' + encodeURIComponent(key);
  }
  /** 人类可读字节大小(B / KB / MB / GB)。 */
  function humanSize(n) {
    if (n == null || isNaN(n)) return '0 B';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + ' MB';
    return (n / 1024 / 1024 / 1024).toFixed(2) + ' GB';
  }
  /** 从 `{error, error_code, error_hint}` 响应里抽取最佳错误描述。 */
  function explainError(body, fallback) {
    if (!body || typeof body !== 'object') return fallback || '未知错误';
    if (body.error_hint) return body.error_hint;
    if (body.error) return body.error;
    return fallback || '未知错误';
  }
  return { $, $$, toast: legacyToast, fmtTime, fmtAbsTime, escapeHtml, highlight, debounce, throttleRaf, mediaUrl, humanSize, explainError };
})();