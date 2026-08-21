/* rs-face Platform · 零依赖 vanilla JS · 模块:api / utils / theme / toast / sidebar / preview / upload / sse / keys / batch / dashboard
 * 优化:虚拟滚动(±6 overscan) · 缩略图 IntersectionObserver 懒加载 · SSE 帧事件 throttleRaf 批渲染
 *     · <img> 全部 loading="lazy" decoding="async" · 无 setInterval 长轮询
 * 增强(v0.2):批量选 + 批量删/归档/导出 · 主题切换(暗/亮/自动) · 智能搜索(正则 + 时间范围)
 *            · 仪表板 · 多级 toast · 确认 modal · a11y + 键盘导航 */
'use strict';

const api = {
  listJobs:    () => fetch('/api/jobs').then(r => r.ok ? r.json().then(j => j.jobs || []) : Promise.reject(new Error(r.status))),
  getJob:      id => fetch('/api/jobs/' + encodeURIComponent(id)).then(r => r.ok ? r.json() : Promise.reject(new Error(r.status))),
  cancelJob:   id => fetch('/api/jobs/' + encodeURIComponent(id) + '/cancel', { method: 'POST' }),
  deleteJob:   id => fetch('/api/jobs/' + encodeURIComponent(id), { method: 'DELETE' }).then(r => r.ok ? r.json() : Promise.reject(new Error(r.status))),
  retryJob:    id => fetch('/api/jobs/' + encodeURIComponent(id) + '/retry', { method: 'POST' }).then(r => r.ok ? r.json() : Promise.reject(new Error(r.status))),
  batch:       (ids, op) => fetch('/api/jobs/batch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids, op }) }).then(r => r.ok ? r.json() : Promise.reject(new Error(r.status))),
  postImage:   file => { const fd = new FormData(); fd.append('file', file); return fetch('/api/jobs/image', { method: 'POST', body: fd }).then(r => r.json()); },
  postVideo:   file => { const fd = new FormData(); fd.append('file', file); return fetch('/api/jobs/video', { method: 'POST', body: fd }).then(r => r.json()); },
  postStream:  url => fetch('/api/jobs/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url }) }).then(r => r.json()),
  getConfig:   () => fetch('/api/config').then(r => r.ok ? r.json() : null).catch(() => null),
  metrics:     () => fetch('/api/metrics').then(r => r.ok ? r.json() : null).catch(() => null),
};

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
  return { $, $$, toast: legacyToast, fmtTime, fmtAbsTime, escapeHtml, highlight, debounce, throttleRaf, mediaUrl };
})();

const state = {
  jobs: [], filter: 'all', algoFilter: '',
  search: '', searchMode: 'plain', searchRange: 'all',
  currentJobId: null, currentJob: null,
  annoVisible: true, eventSource: null, faceCardObserver: null,
  listScrollEl: null, listVpEl: null, listSpacerEl: null,
  itemHeight: 60, itemGap: 6,
  batchMode: false, selectedIds: new Set(),
  themePref: 'auto',
  deletedIds: new Set(),
  config: null,
  // 人脸筛选
  faceScoreMin: 0,
  faceSort: 'time',
  // 性能采样
  _fpsSamples: [],
  _fpsLastFrames: 0,
  _fpsLastAt: 0,
  // KPI 拉取缓存(避免每 200ms 重画)
  _kpi: null,
  _kpiAt: 0,
};

// 持久化 face filter & anno toggle 到 localStorage(避免刷新重置)
const prefStore = (() => {
  const KEYS = {
    faceScoreMin: 'rsface.face.scoreMin',
    faceSort: 'rsface.face.sort',
    annoVisible: 'rsface.annoVisible',
    searchMode: 'rsface.search.mode',
    searchRange: 'rsface.search.range',
  };
  function read(key, fallback) {
    try { const v = localStorage.getItem(key); return v == null ? fallback : v; } catch { return fallback; }
  }
  function write(key, val) { try { localStorage.setItem(key, String(val)); } catch {} }
  function applyAll() {
    const sm = read(KEYS.faceScoreMin, null);
    if (sm != null) state.faceScoreMin = Math.max(0, Math.min(1, parseFloat(sm) || 0));
    const sort = read(KEYS.faceSort, null);
    if (sort && ['time', 'score-desc', 'score-asc', 'size-desc'].indexOf(sort) >= 0) state.faceSort = sort;
    const av = read(KEYS.annoVisible, null);
    if (av != null) state.annoVisible = av === '1';
    const mode = read(KEYS.searchMode, null);
    if (mode && ['plain', 'regex'].indexOf(mode) >= 0) state.searchMode = mode;
    const range = read(KEYS.searchRange, null);
    if (range && ['all', '1h', '24h', '7d', '30d'].indexOf(range) >= 0) state.searchRange = range;
  }
  return { KEYS, read, write, applyAll };
})();
// 立即应用持久化值,确保 sidebar/preview 拿到的是用户上次的状态
prefStore.applyAll();

const theme = (() => {
  const KEY = 'rsface.theme';
  /** 按 pref 返回按钮图标 ◐ / ● / ○ */
  function iconFor(pref) {
    if (pref === 'dark') return '●';
    if (pref === 'light') return '○';
    return '◐'; // auto
  }
  function readPref() { try { return localStorage.getItem(KEY) || 'auto'; } catch { return 'auto'; } }
  function writePref(v) { try { localStorage.setItem(KEY, v); } catch {} }
  function applyPref(pref) {
    state.themePref = pref;
    let effective = pref;
    if (pref === 'auto') {
      effective = (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) ? 'light' : 'dark';
    }
    document.documentElement.dataset.theme = effective;
    utils.$$('#set-theme .seg-btn').forEach(b => {
      const on = b.dataset.theme === pref;
      b.classList.toggle('active', on); b.setAttribute('aria-checked', on ? 'true' : 'false');
    });
    const btn = utils.$('#tb-theme');
    if (btn) {
      btn.textContent = iconFor(pref);
      btn.dataset.mode = pref;
      btn.title = pref === 'auto' ? '主题:自动(点击切换)' : `主题:${pref === 'dark' ? '暗' : '亮'}(点击切换)`;
    }
  }
  /** 顶栏按钮:auto → dark → light → auto 循环 */
  function cycle() {
    const cur = state.themePref || 'auto';
    const next = cur === 'auto' ? 'dark' : cur === 'dark' ? 'light' : 'auto';
    applyPref(next); writePref(next);
  }
  function init() {
    state.themePref = readPref();
    applyPref(state.themePref);
    utils.$$('#set-theme .seg-btn').forEach(b => b.addEventListener('click', () => {
      applyPref(b.dataset.theme); writePref(b.dataset.theme);
    }));
    const tb = utils.$('#tb-theme');
    if (tb) tb.addEventListener('click', cycle);
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => {
        if (state.themePref === 'auto') applyPref('auto');
      });
    }
  }
  return { init, applyPref, cycle };
})();

const toast = (() => {
  const MAX_VISIBLE = 4; // 超过则折叠成 "+N more"
  const host = () => utils.$('#toast-host');
  const queue = []; // {msg, kind, expireAt}
  const livingEls = new Set(); // 当前显示的 DOM

  function ensurePill() {
    const h = host(); if (!h) return null;
    let pill = h.querySelector('.toast-pill');
    if (!pill) {
      pill = document.createElement('div');
      pill.className = 'toast-item toast-pill';
      pill.setAttribute('role', 'status');
      const msgEl = document.createElement('span'); msgEl.className = 'toast-msg pill-msg';
      const closeEl = document.createElement('button'); closeEl.className = 'toast-x'; closeEl.setAttribute('aria-label', '展开'); closeEl.textContent = '↗';
      pill.appendChild(msgEl); pill.appendChild(closeEl);
      // 默认隐藏
      pill.style.display = 'none';
      h.appendChild(pill);
      closeEl.addEventListener('click', () => {
        // 点击展开:把所有 queue 中未过期项立即显示
        const now = Date.now();
        const next = queue.filter(q => q.expireAt > now);
        queue.length = 0;
        next.forEach(q => show(q.msg, q.kind, q.expireAt - now));
        pill.style.display = 'none';
      });
    }
    return pill;
  }

  function refreshPill() {
    const pill = ensurePill(); if (!pill) return;
    const msgEl = pill.querySelector('.pill-msg');
    if (queue.length > 0) {
      msgEl.textContent = `⊕ ${queue.length} 条已折叠`;
      pill.style.display = '';
    } else {
      pill.style.display = 'none';
    }
  }

  function show(msg, kind = 'info', ms = 3500) {
    const h = host(); if (!h) return;
    // 超过可见上限 → 入队列 + 维护 pill
    if (livingEls.size >= MAX_VISIBLE) {
      queue.push({ msg, kind, expireAt: Date.now() + ms });
      refreshPill();
      return;
    }
    const el = document.createElement('div');
    el.className = 'toast-item ' + kind;
    el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
    const msgEl = document.createElement('span'); msgEl.className = 'toast-msg'; msgEl.textContent = msg;
    const closeEl = document.createElement('button'); closeEl.className = 'toast-x'; closeEl.setAttribute('aria-label', '关闭'); closeEl.textContent = '×';
    el.appendChild(msgEl); el.appendChild(closeEl);
    h.appendChild(el);
    livingEls.add(el);
    const close = () => {
      el.classList.add('leaving');
      setTimeout(() => { el.remove(); livingEls.delete(el); refreshPill(); }, 200);
    };
    closeEl.addEventListener('click', close);
    setTimeout(close, ms);
  }
  return {
    info:    (m, ms) => show(m, 'info', ms || 3000),
    success: (m, ms) => show(m, 'success', ms || 3000),
    warn:    (m, ms) => show(m, 'warn', ms || 8000),
    error:   (m, ms) => show(m, 'error', ms || 5000),
  };
})();

const confirmModal = (() => {
  let _resolve = null;
  function open(title, msg) {
    const m = utils.$('#modal-confirm'); if (!m) return Promise.resolve(false);
    utils.$('#modal-confirm-title').textContent = title || '确认';
    utils.$('#modal-confirm-msg').textContent = msg || '确定?';
    m.classList.remove('hidden');
    if (_resolve) _resolve(false);
    return new Promise(res => { _resolve = res; });
  }
  function close(result) {
    const m = utils.$('#modal-confirm'); if (m) m.classList.add('hidden');
    if (_resolve) { const r = _resolve; _resolve = null; r(result); }
  }
  function init() {
    utils.$('#confirm-ok').addEventListener('click', () => close(true));
    utils.$('#confirm-cancel').addEventListener('click', () => close(false));
  }
  return { open, close, init };
})();

const batch = (() => {
  function isActive() { return state.batchMode; }
  function enter() {
    state.batchMode = true; state.selectedIds = new Set();
    utils.$('#sb-batch').classList.remove('hidden');
    utils.$('#sb-scroll').classList.add('batch-on');
    refreshBar();
  }
  function exit() {
    state.batchMode = false; state.selectedIds = new Set();
    utils.$('#sb-batch').classList.add('hidden');
    utils.$('#sb-scroll').classList.remove('batch-on');
    utils.$$('.sb-item').forEach(el => el.classList.remove('selected'));
  }
  function refreshBar() {
    const n = state.selectedIds.size;
    utils.$('#sb-batch-count').textContent = `已选 ${n} 个`;
    utils.$$('.sb-item').forEach(el => el.classList.toggle('selected', state.selectedIds.has(el.id.replace(/^sb-/, ''))));
  }
  function toggle(id) {
    if (state.selectedIds.has(id)) state.selectedIds.delete(id); else state.selectedIds.add(id);
    if (state.selectedIds.size === 0) exit(); else refreshBar();
  }
  async function deleteSelected() {
    const ids = Array.from(state.selectedIds);
    if (!ids.length) return;
    if (!await confirmModal.open('批量删除', `确认删除 ${ids.length} 个任务?此操作不可恢复。`)) return;
    try {
      const r = await api.batch(ids, 'delete');
      ids.forEach(id => state.deletedIds.add(id));
      toast.success(`已删除 ${r.removed_in_mem || ids.length} 个任务`);
      exit(); sidebar.render();
    } catch (e) { toast.error('删除失败: ' + e.message); }
  }
  async function archiveSelected() {
    const ids = Array.from(state.selectedIds);
    if (!ids.length) return;
    try {
      const r = await api.batch(ids, 'archive');
      toast.success(`已归档 ${r.archived} 个任务`);
      exit(); sidebar.render();
    } catch (e) { toast.error('归档失败: ' + e.message); }
  }
  async function exportSelected() {
    const ids = Array.from(state.selectedIds);
    if (!ids.length) return;
    try {
      const r = await api.batch(ids, 'export');
      downloadBlob(JSON.stringify(r.jobs, null, 2), `jobs-batch-${Date.now()}.json`, 'application/json');
      toast.success(`已导出 ${r.jobs.length} 个任务`);
      exit();
    } catch (e) { toast.error('导出失败: ' + e.message); }
  }
  function init() {
    utils.$('#sb-batch-clear').addEventListener('click', exit);
    utils.$('#sb-batch-delete').addEventListener('click', deleteSelected);
    utils.$('#sb-batch-archive').addEventListener('click', archiveSelected);
    utils.$('#sb-batch-export').addEventListener('click', exportSelected);
  }
  return { isActive, enter, exit, toggle, init };
})();

// 平台 KPI:每 2s 拉 /api/metrics,渲染顶栏。带 fps 折线图。
const kpi = (() => {
  const _fpsHist = [];  // 最近 30 个 live_fps_max 样本(平台级)
  const _fpsHistMax = 30;

  function setText(id, txt) {
    const el = utils.$('#' + id);
    if (el) el.textContent = txt;
  }
  function drawSpark(canvas, data, maxOverride) {
    if (!canvas) return;
    const w = canvas.width, h = canvas.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    if (!data || data.length < 2) return;
    const max = (maxOverride && maxOverride > 0) ? maxOverride : Math.max(1, ...data);
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#4fc3f7';
    ctx.beginPath();
    const step = w / (data.length - 1);
    for (let i = 0; i < data.length; i++) {
      const x = i * step;
      const y = h - (data[i] / max) * (h - 2) - 1;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // 渐变填充
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = ctx.strokeStyle + '22';
    ctx.fill();
  }

  async function refresh() {
    const m = await api.metrics().catch(() => null);
    if (!m) return;
    state._kpi = m;
    state._kpiAt = Date.now();

    // running
    setText('kpi-running-v', String(m.running || 0));
    setText('kpi-running-sub', '/ ' + (m.max_concurrency || 0) + ' 并发');

    // fps
    const fps = m.live_fps_max || 0;
    setText('kpi-fps-v', fps > 0 ? fps.toFixed(1) : '—');
    _fpsHist.push(fps);
    if (_fpsHist.length > _fpsHistMax) _fpsHist.shift();
    drawSpark(utils.$('#kpi-fps-spark'), _fpsHist);

    // gpu
    const gpuPct = Math.max(0, Math.min(100, Math.round(m.gpu_pct || 0)));
    setText('kpi-gpu-v', gpuPct + '%');
    const bar = utils.$('#kpi-gpu-bar');
    if (bar) bar.style.width = gpuPct + '%';
    const gpuEl = utils.$('#kpi-gpu');
    if (gpuEl) {
      gpuEl.classList.toggle('kpi-warn', gpuPct === 0 && (m.total_gpu_levels + m.total_cpu_levels) > 0);
    }

    // frames
    setText('kpi-frames-v', String(m.total_frames_processed || 0));
    setText('kpi-frames-sub', '含脸 ' + (m.total_frames_with_face || 0));

    // detections
    setText('kpi-det-v', String(m.total_detections || 0));

    // pass rate
    const pr = (m.cascade_pass_rate || 0);
    setText('kpi-rate-v', pr > 0 ? (pr * 100).toFixed(2) + '%' : '—');

    // failures
    setText('kpi-fail-v', String(m.errored || 0));
    setText('kpi-fail-sub', '取消 ' + (m.cancelled || 0));

    // mode
    setText('kpi-mode-v', (m.mode || '—').toUpperCase());
  }

  function init() {
    refresh();
    setInterval(refresh, 2000);
  }

  return { init, refresh };
})();

const sidebar = (() => {
  const ROW_H = state.itemHeight + state.itemGap;

  function init() {
    state.listScrollEl = utils.$('#sb-scroll');
    state.listVpEl = utils.$('#sb-vp');
    state.listSpacerEl = utils.$('#sb-spacer');
    state.faceCardObserver = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          const img = e.target, src = img.dataset.src;
          if (src) { img.src = src; img.removeAttribute('data-src'); }
          state.faceCardObserver.unobserve(img);
        }
      }
    }, { root: state.listScrollEl, rootMargin: '50px' });
    state.listScrollEl.addEventListener('scroll', renderVp, { passive: true });
    window.addEventListener('resize', utils.throttleRaf(renderVp));
    utils.$$('#sb-filters .sb-filter').forEach(b => b.addEventListener('click', () => {
      state.filter = b.dataset.f;
      utils.$$('#sb-filters .sb-filter').forEach(x => {
        const on = x === b;
        x.classList.toggle('active', on); x.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      render();
    }));
    utils.$$('#sb-algofilter .sb-algo-chip').forEach(b => b.addEventListener('click', () => {
      state.algoFilter = b.dataset.algo || '';
      utils.$$('#sb-algofilter .sb-algo-chip').forEach(x => {
        const on = x === b;
        x.classList.toggle('active', on); x.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
      render();
    }));
    const search = utils.$('#tb-search');
    const onSearch = utils.debounce(() => {
      state.search = (search.value || '').trim();
      utils.$('#tb-search-clear').classList.toggle('hidden', !state.search);
      render();
    }, 120);
    search.addEventListener('input', onSearch);
    utils.$('#tb-search-clear').addEventListener('click', () => {
      search.value = ''; state.search = '';
      utils.$('#tb-search-clear').classList.add('hidden');
      render(); search.focus();
    });
  }

  function rangeMs() {
    if (state.searchRange === 'all') return 0;
    const now = Date.now();
    if (state.searchRange === '1h')  return now - 60 * 60 * 1000;
    if (state.searchRange === '24h') return now - 24 * 60 * 60 * 1000;
    if (state.searchRange === '7d')  return now - 7  * 24 * 60 * 60 * 1000;
    if (state.searchRange === '30d') return now - 30 * 24 * 60 * 60 * 1000;
    return 0;
  }

  function filtered() {
    let jobs = (state.jobs || []).filter(j => !state.deletedIds.has(j.id));
    if (state.filter === 'running') jobs = jobs.filter(j => j.status === 'running' || j.status === 'queued');
    else if (state.filter === 'done') jobs = jobs.filter(j => j.status === 'done');
    else if (state.filter === 'error') jobs = jobs.filter(j => j.status === 'error' || j.status === 'cancelled');
    if (state.algoFilter) {
      // algo 字段由后端在 detector 构建时写入,job 摘要里有 j.algo。
      // 没记录的旧 job (eg. 服务重启前的) 不出现在任何算法 chip 的过滤里。
      const want = state.algoFilter;
      jobs = jobs.filter(j => j.algo === want);
    }
    const since = rangeMs();
    if (since > 0) jobs = jobs.filter(j => (j.created_ms || 0) >= since);
    if (state.search) {
      const q = state.search;
      if (state.searchMode === 'regex') {
        let re;
        try { re = new RegExp(q, 'i'); }
        catch { re = new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i'); }
        jobs = jobs.filter(j => re.test(j.display_name || '') || re.test(j.id || ''));
      } else {
        const ql = q.toLowerCase();
        jobs = jobs.filter(j => (j.display_name || '').toLowerCase().includes(ql) || (j.id || '').toLowerCase().includes(ql));
      }
    }
    return jobs;
  }

  function setJobs(jobs) { state.jobs = jobs; render(); }

  function upsertJob(j) {
    if (state.deletedIds.has(j.id)) return;
    const idx = state.jobs.findIndex(x => x.id === j.id);
    if (idx >= 0) state.jobs[idx] = { ...state.jobs[idx], ...j }; else state.jobs.unshift(j);
    render();
  }

  function render() {
    const list = filtered(), total = list.length;
    utils.$('#sb-foot').textContent = `${total} 匹配 / ${state.jobs.length} 总`;
    state.listSpacerEl.style.height = (total * ROW_H) + 'px';
    updateFilterCounts();
    if (state.currentJobId) {
      const i = list.findIndex(j => j.id === state.currentJobId);
      if (i >= 0) {
        const targetTop = i * ROW_H - state.listScrollEl.clientHeight / 2 + ROW_H / 2;
        const cur = state.listScrollEl.scrollTop;
        if (Math.abs(cur - targetTop) > ROW_H) state.listScrollEl.scrollTop = Math.max(0, targetTop);
      }
    }
    renderVp();
  }

  /** 根据未过滤的 jobs 计算每个 tab 的实时数量(忽略搜索但遵循时间范围)。 */
  function updateFilterCounts() {
    const buckets = { all: 0, running: 0, done: 0, error: 0 };
    const since = rangeMs();
    for (const j of (state.jobs || [])) {
      if (state.deletedIds.has(j.id)) continue;
      if (since > 0 && (j.created_ms || 0) < since) continue;
      buckets.all++;
      const st = j.status;
      if (st === 'running' || st === 'queued') buckets.running++;
      else if (st === 'done') buckets.done++;
      else if (st === 'error' || st === 'cancelled') buckets.error++;
    }
    for (const k of Object.keys(buckets)) {
      const el = document.querySelector(`.sb-filter-count[data-count="${k}"]`);
      if (el) el.textContent = String(buckets[k]);
    }
  }

  function renderVp() {
    const list = filtered(), total = list.length;
    const vp = state.listVpEl, scrollTop = state.listScrollEl.scrollTop, vpH = state.listScrollEl.clientHeight;
    const overscan = 6;
    const start = Math.max(0, Math.floor((scrollTop - overscan * ROW_H) / ROW_H));
    const end = Math.min(total, Math.ceil((scrollTop + vpH + overscan * ROW_H) / ROW_H));
    const keep = new Set();
    for (let i = start; i < end; i++) {
      const j = list[i], id = 'sb-' + j.id;
      keep.add(id);
      let el = vp.querySelector('#' + id);
      if (!el) { el = makeItem(j); vp.appendChild(el); } else updateItem(el, j);
      el.style.transform = `translateY(${i * ROW_H}px)`;
    }
    Array.from(vp.children).forEach(c => { if (!keep.has(c.id)) c.remove(); });
    if (total === 0) {
      if (!vp.querySelector('.sb-empty')) {
        const e = document.createElement('div');
        e.className = 'sb-empty';
        e.innerHTML = '<div class="sb-empty-mark">◐</div><div>暂无任务</div><div style="margin-top:8px">点 + 新建开始识别</div>';
        vp.appendChild(e);
      }
    } else { const e = vp.querySelector('.sb-empty'); if (e) e.remove(); }
  }

  function makeItem(j) {
    const el = document.createElement('div');
    el.id = 'sb-' + j.id; el.className = 'sb-item';
    el.setAttribute('role', 'option');
    if (j.id === state.currentJobId) el.classList.add('active');
    if (state.selectedIds.has(j.id)) el.classList.add('selected');
    el.tabIndex = -1;
    el.addEventListener('click', (e) => {
      if (batch.isActive() || e.shiftKey) { batch.toggle(j.id); return; }
      preview.open(j.id);
    });
    updateItem(el, j);
    return el;
  }

  function updateItem(el, j) {
    const st = j.stats || {}, fp = st.frames_processed || 0, fc = j.face_count || 0;
    // For video/stream jobs prefer `cover_key` (first annotated frame as PNG)
    // over `original_key` which points to the raw .mp4 and would render as
    // a broken <img>. For image jobs both fields are the same.
    const thumbSrc = (j.cover_key || j.original_key) ? utils.mediaUrl(j.cover_key || j.original_key) : null;
    let thumbHtml;
    if (j.status === 'running' || j.status === 'queued') thumbHtml = `<div class="sb-thumb"><div style="opacity:.6">⏳</div></div>`;
    else if (thumbSrc) thumbHtml = `<div class="sb-thumb"><img data-src="${utils.escapeHtml(thumbSrc)}" alt=""></div>`;
    else thumbHtml = `<div class="sb-thumb"><div>·</div></div>`;
    el.innerHTML = `
      <div class="sb-check" aria-hidden="true"><input type="checkbox" ${state.selectedIds.has(j.id) ? 'checked' : ''} tabindex="-1"></div>
      ${thumbHtml}
      <div class="sb-info">
        <div class="sb-name" title="${utils.escapeHtml(j.display_name || j.id)}">${utils.highlight(j.display_name || j.id, state.search)}</div>
        <div class="sb-meta">
          <span class="sb-dot ${j.status || ''}"></span>
          <span>${j.status || '·'}</span><span>·</span>
          <span>${fp} 帧</span><span>·</span>
          <span>${fc} 脸</span>
        </div>
      </div>`;
    el.classList.toggle('selected', state.selectedIds.has(j.id));
    const img = el.querySelector('img[data-src]');
    if (img) state.faceCardObserver.observe(img);
  }

  function setActive(id) {
    utils.$$('.sb-item').forEach(e => {
      const on = e.id === 'sb-' + id;
      e.classList.toggle('active', on);
    });
    // 清除旧的键盘焦点环
    utils.$$('.sb-item.kbd-focus').forEach(e => e.classList.remove('kbd-focus'));
    // 在新激活项上短暂展示 kbd-focus 环(键盘可达性提示)
    if (id) {
      const newEl = utils.$('#sb-' + id);
      if (newEl) {
        newEl.classList.add('kbd-focus');
        clearTimeout(setActive._t);
        setActive._t = setTimeout(() => newEl.classList.remove('kbd-focus'), 900);
      }
    }
    // 自动滚动到可见区域(键盘导航必需)
    if (id) {
      const el = utils.$('#sb-' + id);
      const scroll = state.listScrollEl;
      if (el && scroll) {
        const top = parseFloat(el.style.transform.replace(/[^\d.-]/g, '') || '0');
        const elTop = isNaN(top) ? 0 : top;
        const elBot = elTop + ROW_H;
        const visTop = scroll.scrollTop, visBot = visTop + scroll.clientHeight;
        if (elTop < visTop) scroll.scrollTop = Math.max(0, elTop - 8);
        else if (elBot > visBot) scroll.scrollTop = elBot - scroll.clientHeight + 8;
      }
    }
  }

  function nextItem(dir) {
    const list = filtered(); if (!list.length) return;
    let i = list.findIndex(j => j.id === state.currentJobId);
    if (i < 0) i = 0; else i = Math.max(0, Math.min(list.length - 1, i + dir));
    preview.open(list[i].id);
  }

  /** 跳到列表指定偏移(Home/End/PageUp/PageDown 用)。offset 为 0 = 第一项,-1 = 最后一项 */
  function gotoOffset(offset) {
    const list = filtered(); if (!list.length) return;
    let i;
    if (offset === 0) i = 0;
    else if (offset < 0) i = list.length - 1;
    else i = Math.min(list.length - 1, Math.max(0, offset));
    preview.open(list[i].id);
  }

  return { init, setJobs, upsertJob, render, setActive, nextItem, gotoOffset, filtered };
})();

const preview = (() => {
  let lastFrameIdx = -1, _lastAnnoFrameIdx = -1;

  const showEmpty = () => { utils.$('#pv-empty').classList.remove('hidden'); utils.$('#pv-detail').classList.add('hidden'); };
  const showDetail = () => { utils.$('#pv-empty').classList.add('hidden'); utils.$('#pv-detail').classList.remove('hidden'); };
  const showHint = t => { const h = utils.$('#pv-stage-hint'); h.textContent = t; h.classList.remove('hidden'); };
  const hideHint = () => utils.$('#pv-stage-hint').classList.add('hidden');

  function resetMedia() {
    const img = utils.$('#pv-img'); if (img) { img.removeAttribute('src'); img.classList.add('hidden'); }
    const vid = utils.$('#pv-vid'); if (vid) { vid.removeAttribute('src'); vid.classList.add('hidden'); }
    const db = utils.$('#pv-double'); if (db) db.classList.add('hidden');
    const banner = utils.$('#pv-error-banner'); if (banner) { banner.classList.add('hidden'); banner.textContent = ''; }
    clearOverlay();
  }

  async function open(jobId) {
    state.currentJobId = jobId; lastFrameIdx = -1; _lastAnnoFrameIdx = -1;
    sidebar.setActive(jobId); showDetail(); showHint('加载中…'); resetMedia(); sse.detach();
    // 同步 URL hash(分享可定位)
    if (typeof hashRouter !== 'undefined') hashRouter.setJob(jobId);
    try {
      const job = await api.getJob(jobId);
      state.currentJob = job; render(job);
      if (job.status === 'running' || job.status === 'queued') sse.attach(jobId);
    } catch (e) { toast.error('加载任务失败: ' + e.message); }
  }

  function close() {
    state.currentJobId = null; state.currentJob = null;
    lastFrameIdx = -1; _lastAnnoFrameIdx = -1;
    sse.detach(); showEmpty(); resetMedia(); sidebar.setActive(null);
    if (typeof hashRouter !== 'undefined') hashRouter.clear();
  }

  function render(job) {
    utils.$('#pv-title').textContent = job.display_name || job.id;
    // id 元素渲染为 `#xxx...` + 可点的"复制链接"按钮
    const idEl = utils.$('#pv-id');
    if (idEl) {
      idEl.classList.add('pv-id');
      const short = '#' + (job.id || '').slice(0, 8);
      idEl.innerHTML = '';
      const text = document.createElement('span');
      text.className = 'mono'; text.textContent = short;
      idEl.appendChild(text);
      const copy = document.createElement('button');
      copy.className = 'pv-share-btn';
      copy.title = '复制此任务的深链';
      copy.textContent = '🔗';
      copy.addEventListener('click', () => {
        const url = location.origin + location.pathname + location.search + '#/job/' + encodeURIComponent(job.id);
        navigator.clipboard.writeText(url).then(
          () => toast.success('已复制深链'),
          () => toast.error('复制失败'),
        );
      });
      idEl.appendChild(copy);
    }
    utils.$('#pv-kind').textContent = job.kind || '?';
    const st = job.stats || {};
    utils.$('#pv-frames').textContent = `${st.frames_processed || 0} / ${job.frame_count || st.frames_processed || 0} 帧`;
    utils.$('#pv-faces').textContent = `${job.face_count || 0} 张脸`;
    utils.$('#pv-time').textContent = utils.fmtAbsTime(job.created_ms);
    const status = job.status || 'queued';
    const statusEl = utils.$('#pv-status'); statusEl.textContent = status; statusEl.className = 'pv-status ' + status;
    utils.$('#pv-dot').className = 'pv-dot ' + status;
    utils.$('#pv-cancel').classList.toggle('hidden', !(status === 'running' || status === 'queued'));
    // Bug 2:error/cancelled 都显示原图 + 顶部红色 banner(带复制/重试按钮)
    const banner = utils.$('#pv-error-banner');
    if (status === 'error' || status === 'cancelled') {
      const label = status === 'cancelled' ? '任务已取消' : '识别失败';
      const msg = job.error ? `${label} · ${job.error}` : label;
      if (banner) {
        banner.classList.remove('hidden');
        const msgEl = banner.querySelector('.pv-error-msg');
        const actEl = banner.querySelector('.pv-error-act');
        if (msgEl) msgEl.textContent = msg;
        if (actEl) {
          actEl.innerHTML = '';
          if (job.error) {
            const copy = document.createElement('button');
            copy.textContent = '复制错误';
            copy.title = '复制错误信息到剪贴板';
            copy.addEventListener('click', () => {
              const text = `[${(job.display_name || job.id)}] ${job.error || label}`;
              navigator.clipboard.writeText(text).then(
                () => toast.success('已复制错误'),
                () => toast.error('复制失败'),
              );
            });
            actEl.appendChild(copy);
          }
          const canRetry = job.original_input || (job.kind === 'image' && job.original_key);
          if (canRetry) {
            const retry = document.createElement('button');
            retry.className = 'primary';
            retry.textContent = '↻ 重跑';
            retry.addEventListener('click', retryCurrent);
            actEl.appendChild(retry);
          }
        }
      }
    } else if (banner) {
      banner.classList.add('hidden');
    }
    const retryBtn = utils.$('#pv-retry');
    const canRetry = job.original_input || (job.kind === 'image' && job.original_key);
    if (retryBtn) retryBtn.classList.toggle('hidden', !canRetry);
    if (job.error && status === 'error') {
      if (state.currentJob && state.currentJob.id === job.id && state.currentJob.status !== 'error') {
        toast.error('任务错误: ' + job.error);
      }
    }
    updatePerfMetrics(job);
    renderDispatch(job);
    if (job.kind === 'image') renderImage(job);
    else if (job.kind === 'video') renderVideo(job);
    else renderStream(job);
    renderProgress(job); renderBreakdown(job); renderFaceGrid(job);
  }

  /**
   * 计算并显示 FPS / 已用时长。FPS 用最近 5 秒的 frames_processed 增量估出,
   * 任务未运行时显式置为 "— fps"。
   */
  function updatePerfMetrics(job) {
    const st = job.stats || {};
    const now = Date.now();
    const fpsEl = utils.$('#pv-fps');
    const elapsedEl = utils.$('#pv-elapsed');
    const frames = st.frames_processed || 0;
    if (job.status === 'running' || job.status === 'queued') {
      if (state._fpsLastAt === 0) {
        state._fpsLastAt = now;
        state._fpsLastFrames = frames;
      }
      const dt = (now - state._fpsLastAt) / 1000;
      const inst = dt > 0 ? Math.max(0, (frames - state._fpsLastFrames) / dt) : 0;
      if (dt >= 0.5) {
        state._fpsSamples.push(inst);
        if (state._fpsSamples.length > 6) state._fpsSamples.shift();
        state._fpsLastAt = now;
        state._fpsLastFrames = frames;
      }
      const fps = state._fpsSamples.length
        ? state._fpsSamples.reduce((a, b) => a + b, 0) / state._fpsSamples.length
        : inst;
      if (fpsEl) { fpsEl.textContent = `${fps.toFixed(1)} fps`; fpsEl.classList.remove('idle'); }
    } else {
      // 终态:用 (frames / elapsed_ms) 算平均 fps
      const ms = st.elapsed_ms || 1;
      const avg = ms > 0 ? (frames * 1000) / ms : 0;
      if (fpsEl) { fpsEl.textContent = avg > 0 ? `${avg.toFixed(1)} fps avg` : '— fps'; fpsEl.classList.toggle('idle', avg <= 0); }
      state._fpsLastAt = 0; state._fpsSamples.length = 0;
    }
    if (elapsedEl) {
      const ms = st.elapsed_ms || 0;
      elapsedEl.textContent = ms < 1000 ? `${ms} ms` : ms < 60000 ? `${(ms/1000).toFixed(1)} s` : `${(ms/60000).toFixed(1)} m`;
    }
  }

  // GPU/CPU dispatch 拆分 + fps 折线(只在有 dispatch 字段的 job 上画)。
  function renderDispatch(job) {
    const wrap = utils.$('#pv-dispatch');
    if (!wrap) return;
    const d = job.dispatch;
    const algo = job.algo || (job.stats && job.stats.algo) || '';
    const algoEl = utils.$('#pv-dispatch-algo');
    if (algoEl) {
      algoEl.textContent = algo ? algo.toUpperCase() : '—';
      algoEl.dataset.algo = algo || '';
    }
    if (!d || (d.gpu_levels + d.cpu_levels) === 0) {
      wrap.classList.add('hidden');
      return;
    }
    wrap.classList.remove('hidden');
    const gpuPct = Math.max(0, Math.min(100, Math.round(d.gpu_pct || 0)));
    const gpuEl = utils.$('#pv-dispatch-gpu');
    const cpuEl = utils.$('#pv-dispatch-cpu');
    if (gpuEl) {
      gpuEl.textContent = `GPU ${d.gpu_levels} (${gpuPct}%)`;
      gpuEl.style.flex = String(d.gpu_levels);
    }
    if (cpuEl) {
      cpuEl.textContent = `CPU ${d.cpu_levels} (${100 - gpuPct}%)`;
      cpuEl.style.flex = String(d.cpu_levels);
    }
    const rate = d.cascade_pass_rate || 0;
    const rateEl = utils.$('#pv-dispatch-rate');
    if (rateEl) {
      rateEl.textContent = rate > 0
        ? `通过率 ${(rate * 100).toFixed(2)}%`
        : `通过率 —`;
    }
    // sparkline: 从 stats.fps_window 读(后端每 500ms 推一次)
    const fw = (job.perf && job.perf.fps_window) || (job.stats && job.stats.fps_window) || [];
    const spark = utils.$('#pv-dispatch-spark');
    if (spark && fw.length > 1) {
      const w = spark.width, h = spark.height;
      const ctx = spark.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, w, h);
        const max = Math.max(1, ...fw);
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#4fc3f7';
        ctx.beginPath();
        const step = w / (fw.length - 1);
        for (let i = 0; i < fw.length; i++) {
          const x = i * step;
          const y = h - (fw[i] / max) * (h - 2) - 1;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    }
  }

  function setBar(id, val, opts) {
    const el = utils.$(id); if (!el) return;
    const v = el.querySelector('.v'); if (!v) return;
    v.textContent = val;
    if (opts && opts.pulse) v.classList.add('pulse'); else v.classList.remove('pulse');
  }

  function renderBreakdown(job) {
    const st = job.stats || {};
    const frames = (job.frames || []).length;
    const processed = st.frames_processed || 0;
    const withFace = st.frames_with_face || 0;
    const det = st.total_detections || job.face_count || 0;
    const rate = processed > 0 ? (withFace / processed * 100) : 0;
    let topScore = 0;
    for (const f of job.frames || []) for (const fa of f.faces || []) if ((fa.score || 0) > topScore) topScore = fa.score;
    const algo = (state.config && (state.config.algo || state.config.mode)) || (job.algo) || 'haar';
    setBar('#pv-bb-det', String(det));
    setBar('#pv-bb-fwf', processed > 0 ? `${withFace} / ${processed}` : '0');
    setBar('#pv-bb-rate', processed > 0 ? `${rate.toFixed(1)}%` : '—');
    setBar('#pv-bb-top', topScore > 0 ? topScore.toFixed(2) : '—');
    setBar('#pv-bb-cluster', processed > 0 ? String(clusterFaces(collectFaces(job)).length) : '0');
    setBar('#pv-bb-algo', algo);
  }

  function collectFaces(job) {
    const out = [];
    for (const f of job.frames || []) for (const fa of f.faces || []) {
      out.push({ ts: f.timestamp_ms, x: fa.x, y: fa.y, w: fa.w, h: fa.h, score: fa.score, key: fa.key, frame: f });
    }
    return out;
  }

  function renderImage(job) {
    const img = utils.$('#pv-img');
    const dbl = utils.$('#pv-double');
    const sc = utils.$('#pv-shared-controls');
    if (!img) return;
    if (dbl) dbl.classList.add('hidden');
    if (sc) sc.classList.add('hidden');
    img.classList.remove('hidden');
    const frames = job.frames || [];
    const first = frames.find(f => f.original_key) || frames[0];
    // 优先用 frame.original_key;没有时降级到 job.original_key(error 时只有这个)
    const src = (first && first.original_key) || job.original_key;
    if (src) {
      img.onload = () => { hideHint(); if (first && first.annotated_key) drawOverlay(job, first, img); };
      img.onerror = () => { showHint('原图加载失败'); clearOverlay(); };
      img.src = utils.mediaUrl(src);
    } else { img.classList.add('hidden'); showHint('无原图'); clearOverlay(); }
  }

  function renderVideo(job) {
    const img = utils.$('#pv-img');
    const dbl = utils.$('#pv-double');
    const sc = utils.$('#pv-shared-controls');
    const orig = utils.$('#pv-orig'), anno = utils.$('#pv-anno');
    if (!dbl || !orig) return;
    img.classList.add('hidden');
    dbl.classList.remove('hidden');
    if (sc) sc.classList.remove('hidden');
    if (!job.original_key) { showHint('无视频源'); return; }
    // Bug 1/4:URL 编码
    const newOrigSrc = utils.mediaUrl(job.original_key);
    if (!orig.src || !orig.src.includes(encodeURIComponent(job.original_key))) orig.src = newOrigSrc;
    // 标注视频:job.annotated_key 是后端合成的 mp4;没有就 fall back 到首帧 poster(立即设置,避免等待 metadata)
    if (job.annotated_key && anno) anno.src = utils.mediaUrl(job.annotated_key);
    else {
      const first = (job.frames || []).find(f => f.annotated_key);
      if (first && anno) anno.poster = utils.mediaUrl(first.annotated_key);
    }
    if (anno) bindSync(orig, anno);
    orig.onloadedmetadata = () => {
      hideHint();
      const first = (job.frames || []).find(f => f.annotated_key);
      if (first && anno) drawOverlayOnAnno(job, first);
      refreshSharedProgress();
    };
    anno.onloadedmetadata = () => { hideHint(); refreshSharedProgress(); };
    orig.ontimeupdate = () => { syncVideoOverlayDual(job); refreshSharedProgress(); };
    anno.ontimeupdate = () => { syncOrigByAnnoTime(job); refreshSharedProgress(); };
  }

  function syncVideoOverlayDual(job) {
    const orig = utils.$('#pv-orig'); if (!orig) return;
    const frames = job.frames || []; if (!frames.length) return;
    const cur = orig.currentTime * 1000; let best = null;
    for (const f of frames) {
      if (f.annotated_key && f.timestamp_ms <= cur + 80) best = f;
      else if (f.timestamp_ms > cur + 80) break;
    }
    if (best && best.index !== _lastAnnoFrameIdx) {
      _lastAnnoFrameIdx = best.index;
      drawOverlayOnAnno(job, best);
    }
  }
  function syncOrigByAnnoTime(job) {
    const anno = utils.$('#pv-anno'), orig = utils.$('#pv-orig');
    if (!anno || !orig) return;
    if (Math.abs(orig.currentTime - anno.currentTime) > 0.4) orig.currentTime = anno.currentTime;
  }
  function drawOverlayOnAnno(job, frame) {
    if (!state.annoVisible) return clearOverlay();
    const anno = utils.$('#pv-anno'), c = utils.$('#pv-overlay'), stage = utils.$('#pv-stage');
    if (!anno || !c || !stage) return;
    const rect = stage.getBoundingClientRect(), m = anno.getBoundingClientRect();
    c.style.left = (m.left - rect.left) + 'px';
    c.style.top = (m.top - rect.top) + 'px';
    c.width = m.width; c.height = m.height;
    const natW = anno.videoWidth || (frame.faces[0] ? frame.faces[0].w * 4 : 640);
    const natH = anno.videoHeight || 360;
    drawBoxes(c, frame, natW, natH);
  }

  function refreshSharedProgress() {
    const o = utils.$('#pv-orig'), a = utils.$('#pv-anno');
    const fill = utils.$('#pv-shared-fill'); if (!fill) return;
    const dur = (o && o.duration) || (a && a.duration) || 0;
    const t = (o && o.currentTime) || (a && a.currentTime) || 0;
    const pct = dur > 0 ? (t / dur) * 100 : 0;
    fill.style.width = pct + '%';
    const handle = utils.$('#pv-shared-handle'); if (handle) handle.style.left = pct + '%';
    const time = utils.$('#pv-shared-time'); if (time) time.textContent = `${utils.fmtTime(t * 1000)} / ${utils.fmtTime(dur * 1000)}`;
    const marks = utils.$('#pv-shared-marks'); if (marks && state.currentJob) {
      marks.innerHTML = '';
      const frames = (state.currentJob.frames || []).filter(f => f.faces && f.faces.length);
      const max = 60; const step = Math.max(1, Math.floor(frames.length / max));
      for (let i = 0; i < frames.length; i += step) {
        const f = frames[i];
        const pctTs = dur > 0 ? (f.timestamp_ms / 1000 / dur) * 100 : 0;
        if (pctTs > 100) continue;
        const m = document.createElement('div');
        m.className = 'pv-shared-mark';
        m.style.left = pctTs + '%';
        m.title = utils.fmtTime(f.timestamp_ms);
        m.addEventListener('click', () => {
          if (o) { o.currentTime = f.timestamp_ms / 1000; o.play().catch(() => {}); }
        });
        marks.appendChild(m);
      }
    }
  }

  // 双 video 互锁同步 play/pause/seeked/ratechange/ended
  function bindSync(a, b) {
    if (a._syncBoundTo === b) return;
    a._syncBoundTo = b; b._syncBoundTo = a;
    let locked = false;
    const mirror = (src, dst) => () => {
      if (locked) return;
      locked = true;
      try {
        if (src.playbackRate && dst.playbackRate !== src.playbackRate) dst.playbackRate = src.playbackRate;
        if (Math.abs((dst.currentTime || 0) - (src.currentTime || 0)) > 0.12) dst.currentTime = src.currentTime;
        if (!src.paused && dst.paused) dst.play().catch(() => {});
        if (src.paused && !dst.paused) dst.pause();
      } finally {
        requestAnimationFrame(() => { locked = false; });
      }
    };
    a.addEventListener('play', mirror(a, b));
    a.addEventListener('pause', mirror(a, b));
    a.addEventListener('seeked', mirror(a, b));
    a.addEventListener('ratechange', mirror(a, b));
    a.addEventListener('ended', mirror(a, b));
    b.addEventListener('play', mirror(b, a));
    b.addEventListener('pause', mirror(b, a));
    b.addEventListener('seeked', mirror(b, a));
    b.addEventListener('ratechange', mirror(b, a));
    b.addEventListener('ended', mirror(b, a));
  }

  function initSharedControls() {
    const play = utils.$('#pv-shared-play');
    const track = utils.$('#pv-shared-track');
    const rate = utils.$('#pv-shared-rate');
    if (play) play.addEventListener('click', () => {
      const o = utils.$('#pv-orig'); if (!o) return;
      if (o.paused) o.play().catch(() => {}); else o.pause();
    });
    if (track) track.addEventListener('click', (e) => {
      const o = utils.$('#pv-orig'); if (!o || !o.duration) return;
      const rect = track.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      o.currentTime = Math.max(0, Math.min(1, x)) * o.duration;
    });
    if (rate) rate.addEventListener('change', () => {
      const v = parseFloat(rate.value) || 1;
      const o = utils.$('#pv-orig'), a = utils.$('#pv-anno');
      if (o) o.playbackRate = v;
      if (a) a.playbackRate = v;
    });
    const o = utils.$('#pv-orig');
    if (o) {
      o.addEventListener('play', () => { const b = utils.$('#pv-shared-play'); if (b) b.textContent = '⏸'; });
      o.addEventListener('pause', () => { const b = utils.$('#pv-shared-play'); if (b) b.textContent = '▶'; });
    }
  }

  function initDivider() {
    const div = utils.$('#pv-divider'); const dbl = utils.$('#pv-double');
    const left = utils.$('#pv-half-orig'); const right = utils.$('#pv-half-anno');
    if (!div || !dbl || !left || !right) return;
    let dragging = false;
    div.addEventListener('pointerdown', (e) => { dragging = true; div.setPointerCapture(e.pointerId); });
    div.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const rect = dbl.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const lpct = Math.max(0.15, Math.min(0.85, x / rect.width));
      left.style.flex = `0 0 calc(${lpct * 100}% - 2px)`;
      right.style.flex = '1 1 0';
    });
    div.addEventListener('pointerup', (e) => { dragging = false; div.releasePointerCapture(e.pointerId); });
  }

  function syncVideoOverlay(job, vid) {
    const frames = job.frames || []; if (!frames.length) return;
    const cur = vid.currentTime * 1000; let best = null;
    for (const f of frames) {
      if (f.annotated_key && f.timestamp_ms <= cur + 80) best = f;
      else if (f.timestamp_ms > cur + 80) break;
    }
    if (best && best.index !== _lastAnnoFrameIdx) {
      _lastAnnoFrameIdx = best.index; drawOverlay(job, best, vid);
    }
  }

  function renderStream(job) {
    const img = utils.$('#pv-img');
    const dbl = utils.$('#pv-double');
    const sc = utils.$('#pv-shared-controls');
    if (!img) return;
    if (dbl) dbl.classList.add('hidden');
    if (sc) sc.classList.add('hidden');
    img.classList.remove('hidden');
    const frames = job.frames || [];
    const last = [...frames].reverse().find(f => f.original_key) || frames[frames.length - 1];
    if (!last) {
      // 没有 frame 时降级到 job.original_key
      if (job.original_key) {
        img.onload = () => hideHint();
        img.onerror = () => showHint('原图加载失败');
        img.src = utils.mediaUrl(job.original_key);
      } else { img.classList.add('hidden'); showHint('等待原始帧…'); clearOverlay(); }
      return;
    }
    if (last.index === lastFrameIdx) return;
    lastFrameIdx = last.index;
    img.onload = () => { hideHint(); drawOverlay(job, last, img); };
    img.onerror = () => { showHint('原图加载失败'); };
    img.src = utils.mediaUrl(last.original_key) + (last.original_key && last.original_key.includes('?') ? '&' : '?') + 'v=' + last.index;
  }

  function clearOverlay() {
    const c = utils.$('#pv-overlay'); if (!c) return;
    c.width = 0; c.height = 0;
    const ctx = c.getContext('2d'); if (ctx) ctx.clearRect(0, 0, c.width, c.height);
  }

  function drawOverlay(job, frame, mediaEl) {
    if (!state.annoVisible) return clearOverlay();
    const c = utils.$('#pv-overlay'), stage = utils.$('#pv-stage');
    if (!c || !stage) return;
    const rect = stage.getBoundingClientRect(), mediaRect = mediaEl.getBoundingClientRect();
    c.style.left = (mediaRect.left - rect.left) + 'px';
    c.style.top = (mediaRect.top - rect.top) + 'px';
    c.width = mediaRect.width; c.height = mediaRect.height;
    drawBoxes(c, frame, mediaEl.naturalWidth || mediaEl.videoWidth, mediaEl.naturalHeight || mediaEl.videoHeight);
  }

  function drawBoxes(canvas, frame, natW, natH) {
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!state.annoVisible) return;
    const faces = (frame && frame.faces) || []; if (!faces.length) return;
    const s = Math.min(canvas.width / natW, canvas.height / natH);
    ctx.lineWidth = 2; ctx.strokeStyle = '#4fc3f7';
    ctx.shadowColor = 'rgba(0,0,0,0.6)'; ctx.shadowBlur = 4;
    for (const f of faces) {
      const x = f.x * s, y = f.y * s, w = f.w * s, h = f.h * s;
      ctx.strokeRect(x, y, w, h);
      const txt = (f.score || 0).toFixed(2);
      ctx.font = '11px ui-monospace, monospace';
      const tw = ctx.measureText(txt).width + 6;
      ctx.fillStyle = '#4fc3f7'; ctx.fillRect(x, y - 14, tw, 14);
      ctx.fillStyle = '#001520'; ctx.fillText(txt, x + 3, y - 3);
    }
  }

  let _resizeRaf = 0;
  window.addEventListener('resize', () => {
    if (_resizeRaf) return;
    _resizeRaf = requestAnimationFrame(() => {
      _resizeRaf = 0;
      if (!state.currentJob) return;
      const job = state.currentJob, img = utils.$('#pv-img'), vid = utils.$('#pv-vid');
      if (job.kind === 'image' || job.kind === 'stream') {
        if (img && !img.classList.contains('hidden')) {
          const f = (job.frames || []).find(fr => fr.original_key && fr.index === lastFrameIdx) || (job.frames || []).find(fr => fr.original_key);
          if (f) drawOverlay(job, f, img);
        }
      } else if (job.kind === 'video') {
        if (vid && !vid.classList.contains('hidden')) {
          const f = (job.frames || []).find(fr => fr.index === _lastAnnoFrameIdx) || (job.frames || []).find(fr => fr.annotated_key);
          if (f) drawOverlay(job, f, vid);
        }
      }
    });
  });

  function renderProgress(job) {
    const st = job.stats || {};
    const processed = st.frames_processed || 0, total = job.frame_count || processed;
    const pct = total > 0 ? (processed / total) * 100 : 0;
    utils.$('#pv-prog-fill').style.width = pct + '%';
    utils.$('#pv-prog-text').textContent = `${processed} / ${total}`;
    utils.$('#pv-prog-pct').textContent = pct.toFixed(0) + '%';
    const marks = utils.$('#pv-prog-marks'); if (marks) marks.innerHTML = '';
    if (total > 0) {
      const max = 80, step = Math.max(1, Math.floor((job.frames || []).length / max));
      let count = 0;
      for (let i = 0; i < (job.frames || []).length; i += step) {
        const f = job.frames[i];
        if (f.faces && f.faces.length) {
          const m = document.createElement('div');
          m.className = 'pv-prog-mark';
          m.style.left = ((f.index / total) * 100) + '%';
          marks.appendChild(m);
          if (++count > max) break;
        }
      }
    }
  }

  function renderFaceGrid(job) {
    const grid = utils.$('#pv-faces-grid');
    const all = [];
    for (const f of job.frames || []) {
      for (const face of (f.faces || [])) {
        all.push({ ts: f.timestamp_ms, key: face.key, score: face.score, frame: f, x: face.x, y: face.y, w: face.w, h: face.h });
      }
    }
    const filtered = all.filter(f => (f.score || 0) >= state.faceScoreMin);
    const sorters = {
      'time':       (a, b) => a.ts - b.ts,
      'score-desc': (a, b) => b.score - a.score,
      'score-asc':  (a, b) => a.score - b.score,
      'size-desc':  (a, b) => (b.w * b.h) - (a.w * a.h),
    };
    filtered.sort(sorters[state.faceSort] || sorters.time);
    const clusters = clusterFaces(filtered);
    const totalClusters = clusterFaces(all).length;
    utils.$('#pv-faces-count').textContent = filtered.length === all.length
      ? (clusters.length ? `${all.length} 张 · ${clusters.length} 组` : '—')
      : `${filtered.length} / ${all.length} 张 · ${clusters.length}/${totalClusters} 组`;
    if (!filtered.length) {
      const msg = all.length ? '当前阈值过滤掉所有结果' : '尚未出现人脸';
      grid.innerHTML = `<div class="empty">${msg}</div>`;
      return;
    }
    const useClusters = state.faceSort === 'time';
    grid.innerHTML = '';
    if (useClusters) {
      for (const cl of clusters) {
        const rep = cl.rep;
        const card = makeFaceCard(rep, cl.members.length);
        card.addEventListener('click', () => seekToFrame(rep.frame));
        // 双击打开 lightbox(单双击共存:单击跳视频,双击看大图)
        card.addEventListener('dblclick', () => lightbox.open(rep.frame));
        grid.appendChild(card);
      }
    } else {
      // 按其他排序时不聚类(否则代表帧错位),逐条渲染
      for (const f of filtered) {
        const card = makeFaceCard(f, 1);
        card.addEventListener('click', () => seekToFrame(f.frame));
        card.addEventListener('dblclick', () => lightbox.open(f.frame));
        grid.appendChild(card);
      }
    }
    grid.querySelectorAll('img[data-src]').forEach(img => state.faceCardObserver.observe(img));
  }

  function makeFaceCard(rep, members) {
    const card = document.createElement('div');
    card.className = 'face-card';
    const idxText = rep.frame ? `#${rep.frame.index}` : '';
    card.innerHTML = `
      <div class="fc-img-wrap">
        <div class="fc-skel"></div>
        <img loading="lazy" decoding="async" data-src="${utils.escapeHtml(utils.mediaUrl(rep.key))}" alt="face">
        ${idxText ? `<div class="fc-idx">${idxText}</div>` : ''}
        <div class="fc-time">⏱ ${utils.fmtTime(rep.ts)}</div>
        ${members > 1 ? `<div class="fc-badge">×${members}</div>` : ''}
      </div>
      <div class="fc-score">score <span>${rep.score.toFixed(2)}</span></div>`;
    return card;
  }

  function seekToFrame(frame) {
    if (!frame) return;
    const job = state.currentJob;
    if (!job) return;
    if (job.kind === 'video') {
      const o = utils.$('#pv-orig') || utils.$('#pv-vid');
      if (o) { o.currentTime = frame.timestamp_ms / 1000; o.play().catch(() => {}); }
    }
  }

  function initFaceFilters() {
    const score = utils.$('#pv-ff-score');
    const scoreVal = utils.$('#pv-ff-score-val');
    const sort = utils.$('#pv-ff-sort');
    // 从持久化值同步到 UI
    if (score) score.value = String(Math.round(state.faceScoreMin * 100));
    if (scoreVal) scoreVal.textContent = state.faceScoreMin.toFixed(2);
    if (sort) sort.value = state.faceSort;
    if (score) score.addEventListener('input', () => {
      state.faceScoreMin = (parseInt(score.value, 10) || 0) / 100;
      if (scoreVal) scoreVal.textContent = state.faceScoreMin.toFixed(2);
      prefStore.write(prefStore.KEYS.faceScoreMin, state.faceScoreMin);
      if (state.currentJob) renderFaceGrid(state.currentJob);
    });
    if (sort) sort.addEventListener('change', () => {
      state.faceSort = sort.value;
      prefStore.write(prefStore.KEYS.faceSort, state.faceSort);
      if (state.currentJob) renderFaceGrid(state.currentJob);
    });
  }

  function clusterFaces(raw) {
    if (!raw.length) return [];
    const DT_MS = 2000, DIST_PX = 140;
    const clusters = []; let cur = [raw[0]];
    for (let i = 1; i < raw.length; i++) {
      const prev = cur[cur.length - 1], f = raw[i], dt = f.ts - prev.ts;
      const pcx = (prev.x || 0) + (prev.w || 0) / 2, pcy = (prev.y || 0) + (prev.h || 0) / 2;
      const fcx = (f.x || 0) + (f.w || 0) / 2, fcy = (f.y || 0) + (f.h || 0) / 2;
      if (dt < DT_MS && Math.hypot(fcx - pcx, fcy - pcy) < DIST_PX) cur.push(f);
      else { clusters.push(cur); cur = [f]; }
    }
    clusters.push(cur);
    return clusters.map(c => ({ rep: c[Math.floor(c.length / 2)], members: c }));
  }

  function toggleAnno() {
    state.annoVisible = !state.annoVisible;
    prefStore.write(prefStore.KEYS.annoVisible, state.annoVisible ? '1' : '0');
    const btn = utils.$('#pv-toggle-anno');
    if (btn) {
      btn.textContent = state.annoVisible ? '标注 ●' : '标注 〇';
      btn.setAttribute('aria-pressed', state.annoVisible ? 'true' : 'false');
    }
    if (state.currentJob) render(state.currentJob);
  }

  return { open, close, render, toggleAnno, showEmpty, showDetail, initSharedControls, initDivider, initFaceFilters, updatePerfMetrics, renderDispatch };
})();

const upload = (() => {
  function init() {
    utils.$('#tb-new').addEventListener('click', openModal);
    utils.$('#tb-settings').addEventListener('click', openSettings);
    utils.$('#tb-help').addEventListener('click', () => utils.$('#modal-help').classList.remove('hidden'));
    utils.$$('.new-type').forEach(b => b.addEventListener('click', () => selectType(b.dataset.type)));
    utils.$$('[data-close]').forEach(el => el.addEventListener('click', closeAllModals));
    setupDropzone('#dz-image', '#file-image', files => {
      if (files.length > 1) for (const f of files) submitImage(f); else submitImage(files[0]);
      closeAllModals();
    });
    setupDropzone('#dz-video', '#file-video', files => { submitVideo(files[0]); closeAllModals(); });
    utils.$('#video-url-go').addEventListener('click', () => {
      const url = utils.$('#video-url').value.trim();
      if (!url) return toast.warn('请输入视频 URL');
      submitStream(url); closeAllModals();
    });
    utils.$('#stream-go').addEventListener('click', () => {
      const url = utils.$('#stream-url').value.trim();
      if (!url) return toast.warn('请输入流地址');
      submitStream(url); closeAllModals();
    });
    utils.$('#tb-search-opts').addEventListener('click', () => {
      utils.$('#modal-search-opts').classList.remove('hidden');
      utils.$$('#search-mode .seg-btn').forEach(b => {
        const on = b.dataset.mode === state.searchMode;
        b.classList.toggle('active', on); b.setAttribute('aria-checked', on ? 'true' : 'false');
      });
      utils.$$('#search-range .seg-btn').forEach(b => {
        const on = b.dataset.range === state.searchRange;
        b.classList.toggle('active', on); b.setAttribute('aria-checked', on ? 'true' : 'false');
      });
    });
    utils.$$('#search-mode .seg-btn').forEach(b => b.addEventListener('click', () => {
      state.searchMode = b.dataset.mode;
      utils.$$('#search-mode .seg-btn').forEach(x => {
        const on = x === b; x.classList.toggle('active', on); x.setAttribute('aria-checked', on ? 'true' : 'false');
      });
      sidebar.render();
    }));
    utils.$$('#search-range .seg-btn').forEach(b => b.addEventListener('click', () => {
      state.searchRange = b.dataset.range;
      utils.$$('#search-range .seg-btn').forEach(x => {
        const on = x === b; x.classList.toggle('active', on); x.setAttribute('aria-checked', on ? 'true' : 'false');
      });
      sidebar.render();
    }));
    utils.$('#tb-dashboard').addEventListener('click', () => { dashboard.open(); });
    setupGlobalDrop(); setupGlobalPaste();
  }

  const openModal = () => { utils.$('#modal-new').classList.remove('hidden'); selectType('image'); };
  const openSettings = () => { utils.$('#modal-settings').classList.remove('hidden'); loadSettings(); };
  const closeAllModals = () => utils.$$('.modal').forEach(m => m.classList.add('hidden'));
  function selectType(t) {
    utils.$$('.new-type').forEach(b => {
      const on = b.dataset.type === t; b.classList.toggle('active', on);
    });
    utils.$$('.new-pane').forEach(p => p.classList.add('hidden'));
    const pane = utils.$('#pane-' + t); if (pane) pane.classList.remove('hidden');
  }
  async function loadSettings() {
    const cfg = await api.getConfig();
    state.config = cfg;
    const setVal = (sel, val) => { const el = utils.$(sel); if (el) el.textContent = val || 'n/a'; };
    if (cfg) {
      setVal('#set-mode', cfg.mode);
      setVal('#set-cnn', cfg.cnn_weights_status || cfg.cnn_weights_path || 'n/a');
      setVal('#set-cascade', cfg.cascade_status || 'n/a');
    } else { setVal('#set-mode', 'n/a (404)'); setVal('#set-cnn', 'n/a'); setVal('#set-cascade', 'n/a'); }
  }
  function setupDropzone(zoneSel, inputSel, handler) {
    const zone = utils.$(zoneSel), input = utils.$(inputSel); if (!zone || !input) return;
    zone.addEventListener('click', e => { if (!e.target.closest('input')) input.click(); });
    input.addEventListener('change', () => { if (input.files.length) handler(Array.from(input.files)); input.value = ''; });
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', e => { if (!zone.contains(e.relatedTarget)) zone.classList.remove('dragover'); });
    zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('dragover'); });
  }
  function setupGlobalDrop() {
    document.addEventListener('dragover', e => { if (e.dataTransfer) e.preventDefault(); });
    document.addEventListener('drop', async e => {
      e.preventDefault();
      if (!utils.$('#modal-new').classList.contains('hidden')) return;
      if (!e.dataTransfer || !e.dataTransfer.files || !e.dataTransfer.files.length) return;
      const files = Array.from(e.dataTransfer.files), t = files[0].type || '';
      if (t.startsWith('image/')) for (const f of files) submitImage(f);
      else if (t.startsWith('video/')) submitVideo(files[0]);
      else toast.warn('不支持的文件类型: ' + t);
    });
  }
  function setupGlobalPaste() {
    document.addEventListener('paste', async e => {
      if (e.target && e.target.matches && e.target.matches('input, textarea')) return;
      const cd = e.clipboardData; if (!cd) return;
      for (const it of cd.items || []) {
        if (it.kind === 'file' && it.type && it.type.startsWith('image/')) {
          const f = it.getAsFile(); if (f) { e.preventDefault(); submitImage(f); return; }
        }
      }
    });
  }
  async function submitImage(file) {
    try {
      toast.info(`上传 ${file.name}…`);
      const data = await api.postImage(file);
      if (data.error) return toast.error('上传失败: ' + data.error);
      toast.success('已提交 #' + (data.job_id || '').slice(0, 8));
      try {
        const job = await api.getJob(data.job_id);
        sidebar.upsertJob(job); preview.open(job.id);
      } catch {}
    } catch (e) { toast.error('上传失败: ' + e.message); }
  }
  async function submitVideo(file) {
    try {
      toast.info(`上传 ${file.name}…`);
      const data = await api.postVideo(file);
      if (data.error) return toast.error('上传失败: ' + data.error);
      toast.success('已提交 #' + (data.job_id || '').slice(0, 8));
      const job = await api.getJob(data.job_id);
      sidebar.upsertJob(job); preview.open(job.id);
    } catch (e) { toast.error('上传失败: ' + e.message); }
  }
  async function submitStream(url) {
    try {
      const data = await api.postStream(url);
      if (data.error) return toast.error('启动失败: ' + data.error);
      toast.success('已启动流 #' + (data.job_id || '').slice(0, 8));
      const job = await api.getJob(data.job_id);
      sidebar.upsertJob(job); preview.open(job.id);
    } catch (e) { toast.error('启动失败: ' + e.message); }
  }
  return { init, openModal, openSettings };
})();

const sse = (() => {
  const scheduleRefresh = utils.throttleRaf(async () => {
    if (!state.currentJobId) return;
    if (state.deletedIds.has(state.currentJobId)) return;
    try {
      const job = await api.getJob(state.currentJobId);
      state.currentJob = job; preview.render(job); sidebar.upsertJob(job);
    } catch {}
  });

  // 把后端 heartbeat 合并到 state.currentJob,触发便宜 DOM 更新,
  // 不再每次都去拉 /api/jobs/:id。
  // 期望 msg = { type:"heartbeat", fps, frames_processed,
  //              gpu_levels, cpu_levels, gpu_pct,
  //              cascade_evals, cascade_passes, fps_window? }
  function applyHeartbeat(msg) {
    const job = state.currentJob;
    if (!job) return;
    if (!job.stats) job.stats = {};
    if (!job.dispatch) job.dispatch = {};
    if (!job.perf) job.perf = { fps_window: [] };
    if (msg.fps !== undefined) {
      job.stats.frames_processed = msg.frames_processed || job.stats.frames_processed || 0;
      const fw = job.perf.fps_window;
      fw.push(msg.fps);
      if (fw.length > 60) fw.shift();
    }
    if (msg.gpu_levels !== undefined) job.dispatch.gpu_levels = msg.gpu_levels;
    if (msg.cpu_levels !== undefined) job.dispatch.cpu_levels = msg.cpu_levels;
    if (msg.gpu_pct     !== undefined) job.dispatch.gpu_pct     = msg.gpu_pct;
    if (msg.cascade_evals !== undefined) job.dispatch.cascade_evals = msg.cascade_evals;
    if (msg.cascade_passes !== undefined) {
      job.dispatch.cascade_passes = msg.cascade_passes;
      const ev = job.dispatch.cascade_evals || 0;
      job.dispatch.cascade_pass_rate = ev > 0 ? job.dispatch.cascade_passes / ev : 0;
    }
    // 局部刷新预览页的 dispatch / fps 单元
    preview.updatePerfMetrics(job);
    preview.renderDispatch(job);
  }

  function attach(jobId) {
    detach();
    if (!('EventSource' in window)) return;
    const es = new EventSource('/api/jobs/' + encodeURIComponent(jobId) + '/events');
    state.eventSource = es;
    es.onmessage = (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === 'frame') scheduleRefresh();
      else if (msg.type === 'heartbeat') applyHeartbeat(msg);
      else if (msg.type === 'done' || msg.type === 'cancelled' || msg.type === 'error') {
        detach(); scheduleRefresh();
        if (msg.type === 'done') toast.success('任务完成');
        else if (msg.type === 'cancelled') toast.info('已停止');
        else toast.error('任务出错: ' + (msg.message || ''));
      }
    };
    es.onerror = () => { /* browser auto-retry */ };
  }
  function detach() { if (state.eventSource) { state.eventSource.close(); state.eventSource = null; } }
  return { attach, detach, applyHeartbeat };
})();

/**
 * 人脸 lightbox:点击人脸卡片或双击主图打开。
 * - 展示当前 frame 的 annotated_key(后端已画好检测框)+ 框选详情
 * - 跨 frame 翻页:同一 job 内所有有脸帧都可导航
 * - 视频任务提供"跳到视频"按钮,定位到该时间戳
 */
const lightbox = (() => {
  const els = {};
  let _frames = [];   // 当前 job 的所有 frame(只含带脸帧)
  let _idx = 0;
  // 缩放/平移状态
  let _scale = 1, _tx = 0, _ty = 0;
  let _activeFace = -1; // 当前高亮的 face index

  function init() {
    els.modal = utils.$('#modal-lightbox');
    els.img = utils.$('#lb-img');
    els.overlay = utils.$('#lb-overlay');
    els.stage = utils.$('#lb-stage');
    els.skel = utils.$('#lb-skel');
    els.meta = utils.$('#lb-meta');
    els.pos = utils.$('#lb-pos');
    els.prev = utils.$('#lb-prev');
    els.next = utils.$('#lb-next');
    els.seek = utils.$('#lb-seek');
    els.zoomReset = utils.$('#lb-zoom-reset');
    els.rail = utils.$('#lb-rail');
    els.railCount = utils.$('#lb-rail-count');
    els.railList = utils.$('#lb-rail-list');
    if (!els.modal) return;
    els.prev && els.prev.addEventListener('click', () => step(-1));
    els.next && els.next.addEventListener('click', () => step(+1));
    els.seek && els.seek.addEventListener('click', () => {
      const f = _frames[_idx];
      if (!f || !state.currentJob || state.currentJob.kind !== 'video') return;
      const o = utils.$('#pv-orig') || utils.$('#pv-vid');
      if (o) { o.currentTime = (f.timestamp_ms || 0) / 1000; o.play().catch(() => {}); }
      close();
    });
    els.zoomReset && els.zoomReset.addEventListener('click', () => resetZoom());
    // 缩放 / 平移
    els.stage && els.stage.addEventListener('wheel', onWheel, { passive: false });
    els.stage && els.stage.addEventListener('pointerdown', onPointerDown);
    // 键盘 ←/→ 翻页,Esc 关闭(已经在全局 initKeys 里统一处理 modal,这里只防重复)
    document.addEventListener('keydown', (e) => {
      if (els.modal.classList.contains('hidden')) return;
      if (e.target && e.target.matches && e.target.matches('input, textarea, [contenteditable]')) return;
      if (e.key === 'ArrowLeft')  { e.preventDefault(); step(-1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); step(+1); }
      if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomAt(1.2, null); }
      if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomAt(1 / 1.2, null); }
      if (e.key === '0') { e.preventDefault(); resetZoom(); }
    });
  }

  function open(frame) {
    const job = state.currentJob; if (!job || !els.modal) return;
    _frames = (job.frames || []).filter(f => f.faces && f.faces.length);
    _idx = Math.max(0, _frames.findIndex(f => f.index === frame.index));
    if (_idx < 0) _idx = 0;
    _activeFace = -1;
    resetZoom();
    showCurrent();
    els.modal.classList.remove('hidden');
  }

  function close() {
    if (els.modal) els.modal.classList.add('hidden');
    resetZoom();
    _activeFace = -1;
  }

  function resetZoom() {
    _scale = 1; _tx = 0; _ty = 0;
    applyTransform();
    if (els.zoomReset) els.zoomReset.hidden = true;
  }

  function applyTransform() {
    if (!els.img) return;
    els.img.style.transform = `translate(${_tx}px, ${_ty}px) scale(${_scale})`;
    if (els.zoomReset) {
      els.zoomReset.hidden = _scale === 1;
      els.zoomReset.textContent = `${Math.round(_scale * 100)}%`;
    }
    // 同步 overlay 位置
    if (els.overlay && els.stage) drawOverlay();
  }

  function onWheel(e) {
    if (els.modal.classList.contains('hidden')) return;
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    zoomAt(factor, { x: e.clientX, y: e.clientY });
  }

  function zoomAt(factor, anchor) {
    const old = _scale;
    const next = Math.max(0.5, Math.min(6, _scale * factor));
    if (next === _scale) return;
    if (anchor && els.stage) {
      const rect = els.stage.getBoundingClientRect();
      const ax = anchor.x - rect.left - rect.width / 2;
      const ay = anchor.y - rect.top - rect.height / 2;
      // 调整 tx/ty 使 anchor 点在缩放后仍对齐
      _tx = ax - (ax - _tx) * (next / old);
      _ty = ay - (ay - _ty) * (next / old);
    }
    _scale = next;
    if (els.img) els.img.classList.add('zooming');
    applyTransform();
    clearTimeout(els.img._t);
    els.img._t = setTimeout(() => els.img && els.img.classList.remove('zooming'), 180);
  }

  function onPointerDown(e) {
    if (els.modal.classList.contains('hidden')) return;
    if (_scale <= 1) return; // 没放大时不启动拖动
    els.stage.setPointerCapture(e.pointerId);
    els.stage.classList.add('dragging');
    const startX = e.clientX, startY = e.clientY;
    const baseTx = _tx, baseTy = _ty;
    const onMove = (ev) => {
      _tx = baseTx + (ev.clientX - startX);
      _ty = baseTy + (ev.clientY - startY);
      applyTransform();
    };
    const onUp = (ev) => {
      els.stage.releasePointerCapture(ev.pointerId);
      els.stage.classList.remove('dragging');
      els.stage.removeEventListener('pointermove', onMove);
      els.stage.removeEventListener('pointerup', onUp);
    };
    els.stage.addEventListener('pointermove', onMove);
    els.stage.addEventListener('pointerup', onUp);
  }

  function drawOverlay() {
    const f = _frames[_idx]; if (!f || !els.overlay || !els.stage) return;
    const c = els.overlay;
    const rect = els.stage.getBoundingClientRect();
    c.width = rect.width; c.height = rect.height;
    c.style.width = rect.width + 'px';
    c.style.height = rect.height + 'px';
    const ctx = c.getContext('2d'); if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);
    const img = els.img;
    if (!img) return;
    const natW = img.naturalWidth || 640;
    const natH = img.naturalHeight || 360;
    // 计算 img 实际绘制区域(object-fit: contain)
    const ar = natW / natH, sar = rect.width / rect.height;
    let dw, dh, dx, dy;
    if (ar > sar) { dw = rect.width; dh = rect.width / ar; dx = 0; dy = (rect.height - dh) / 2; }
    else          { dh = rect.height; dw = rect.height * ar; dy = 0; dx = (rect.width - dw) / 2; }
    const s = dw / natW;
    const faces = f.faces || [];
    for (let i = 0; i < faces.length; i++) {
      const fa = faces[i];
      const x = dx + (fa.x || 0) * s, y = dy + (fa.y || 0) * s;
      const w = (fa.w || 0) * s, h = (fa.h || 0) * s;
      const isActive = i === _activeFace;
      ctx.lineWidth = isActive ? 3 : 2;
      ctx.strokeStyle = isActive ? '#f0c674' : '#4fc3f7';
      ctx.shadowColor = 'rgba(0, 0, 0, 0.55)'; ctx.shadowBlur = isActive ? 6 : 4;
      ctx.strokeRect(x, y, w, h);
      const txt = (fa.score || 0).toFixed(2);
      ctx.font = `${isActive ? 'bold ' : ''}12px ui-monospace, monospace`;
      const tw = ctx.measureText(txt).width + 8;
      ctx.fillStyle = isActive ? '#f0c674' : '#4fc3f7';
      ctx.fillRect(x, y - 16, tw, 16);
      ctx.fillStyle = '#001520'; ctx.fillText(txt, x + 4, y - 4);
    }
  }

  function showCurrent() {
    const f = _frames[_idx]; if (!f) return;
    _activeFace = -1;
    const img = els.img, skel = els.skel;
    if (skel) skel.style.display = '';
    const src = f.annotated_key || f.original_key;
    if (img) {
      img.onload = () => {
        if (skel) skel.style.display = 'none';
        applyTransform();
      };
      img.onerror = () => {
        if (skel) skel.style.display = 'none';
        toast.warn('标注帧加载失败');
      };
      img.src = utils.mediaUrl(src);
    }
    if (els.meta) {
      const ts = utils.fmtTime(f.timestamp_ms || 0);
      const n = (f.faces || []).length;
      const topScore = Math.max(0, ...(f.faces || []).map(x => x.score || 0));
      els.meta.textContent = `#${f.index} · ${ts} · ${n} 张脸 · 最高分 ${topScore.toFixed(2)}`;
    }
    if (els.pos) els.pos.textContent = `${_idx + 1} / ${_frames.length}`;
    if (els.prev) els.prev.disabled = _idx <= 0;
    if (els.next) els.next.disabled = _idx >= _frames.length - 1;
    const job = state.currentJob;
    if (els.seek) els.seek.classList.toggle('hidden', !job || job.kind !== 'video');
    // 右侧人脸列表面板
    renderRail(f);
  }

  function renderRail(f) {
    if (!els.railList) return;
    const faces = f.faces || [];
    if (els.railCount) els.railCount.textContent = String(faces.length);
    els.railList.innerHTML = '';
    if (!faces.length) {
      els.railList.innerHTML = '<div class="lb-rail-item" style="color:var(--fg-dim);cursor:default">本帧无人脸</div>';
      return;
    }
    faces.forEach((fa, i) => {
      const it = document.createElement('div');
      it.className = 'lb-rail-item' + (i === _activeFace ? ' active' : '');
      it.innerHTML = `<span class="lb-rail-i">#${i + 1}</span><span class="lb-rail-s">${(fa.score || 0).toFixed(2)}</span><span style="color:var(--fg-dim);font-size:10px">${fa.w}×${fa.h}</span>`;
      it.addEventListener('click', () => {
        _activeFace = (_activeFace === i) ? -1 : i;
        renderRail(f);
        drawOverlay();
      });
      els.railList.appendChild(it);
    });
  }

  function step(d) {
    const n = _idx + d;
    if (n < 0 || n >= _frames.length) return;
    _idx = n; showCurrent();
  }

  return { init, open, close };
})();

const dashboard = (() => {
  async function compute() {
    const jobs = await api.listJobs();
    const total = jobs.length;
    let running = 0, done = 0, err = 0, faces = 0, ms = 0;
    const algoCount = {};
    const buckets = new Array(24).fill(0);
    const now = Date.now();
    for (const j of jobs) {
      if (j.status === 'running' || j.status === 'queued') running++;
      else if (j.status === 'done') done++;
      else if (j.status === 'error' || j.status === 'cancelled') err++;
      faces += j.face_count || 0;
      ms += (j.stats && j.stats.elapsed_ms) || 0;
      const algo = (j.kind || 'image');
      algoCount[algo] = (algoCount[algo] || 0) + (j.face_count || 1);
      if (j.created_ms) {
        const hoursAgo = Math.floor((now - j.created_ms) / (60 * 60 * 1000));
        if (hoursAgo >= 0 && hoursAgo < 24) buckets[23 - hoursAgo] += 1;
      }
    }
    if (state.config && Array.isArray(state.config.available_algos)) {
      for (const a of state.config.available_algos) {
        if (!(a in algoCount)) algoCount[a] = 0;
      }
    }
    return { total, running, done, err, faces, ms, algoCount, buckets };
  }

  function renderStats(d) {
    const fmt = ms => ms < 1000 ? `${ms} ms` : ms < 60000 ? `${(ms/1000).toFixed(1)} s` : `${(ms/60000).toFixed(1)} m`;
    const tiles = [
      { k: '总任务', v: d.total, c: 'accent' },
      { k: '进行中', v: d.running, c: 'warn' },
      { k: '已完成', v: d.done, c: 'success' },
      { k: '错误', v: d.err, c: 'danger' },
      { k: '人脸总数', v: d.faces, c: 'accent' },
      { k: '总检测时长', v: fmt(d.ms), c: 'fg' },
    ];
    utils.$('#dash-stats').innerHTML = tiles.map(t =>
      `<div class="dash-tile ${t.c}"><div class="dash-tile-v">${t.v}</div><div class="dash-tile-k">${t.k}</div></div>`
    ).join('');
  }
  function renderAlgos(d) {
    const entries = Object.entries(d.algoCount).sort((a,b) => b[1]-a[1]);
    const max = Math.max(1, ...entries.map(e => e[1]));
    const el = utils.$('#dash-algos');
    if (!entries.length) { el.innerHTML = '<div class="hint">暂无数据</div>'; return; }
    el.innerHTML = entries.map(([k,v]) => {
      const pct = (v / max * 100).toFixed(1);
      return `<div class="dash-bar-row">
        <div class="dash-bar-label">${utils.escapeHtml(k)}</div>
        <div class="dash-bar"><div class="dash-bar-fill" style="width:${pct}%"></div><span class="dash-bar-val">${v}</span></div>
      </div>`;
    }).join('');
  }
  function renderTimeline(d) {
    const max = Math.max(1, ...d.buckets);
    const W = 460, H = 80, bw = W / d.buckets.length;
    let bars = '', labels = '';
    d.buckets.forEach((v, i) => {
      const h = (v / max) * (H - 14);
      const x = i * bw + 1, y = H - h - 10, w = bw - 2;
      bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="1.5"><title>${i}:00 — ${v} jobs</title></rect>`;
    });
    const lblHrs = ['现在', '-12h', '-24h'];
    const lblX = [W - 28, W/2 - 12, 0];
    lblHrs.forEach((t, i) => { labels += `<text x="${lblX[i].toFixed(0)}" y="${H + 2}">${t}</text>`; });
    utils.$('#dash-timeline').innerHTML =
      `<svg viewBox="0 0 ${W} ${H + 6}" class="dash-svg" preserveAspectRatio="none">${bars}${labels}</svg>`;
  }

  async function open() {
    utils.$('#modal-dashboard').classList.remove('hidden');
    utils.$('#dash-stats').innerHTML = '<div class="hint">加载中…</div>';
    try {
      const d = await compute();
      renderStats(d); renderAlgos(d); renderTimeline(d);
      utils.$('#dash-meta').textContent = `数据源:内存中的 /api/jobs · 任务总数 ${d.total} · 刷新于 ${new Date().toLocaleTimeString()}`;
    } catch (e) {
      utils.$('#dash-stats').innerHTML = '<div class="hint">加载失败: ' + utils.escapeHtml(e.message) + '</div>';
    }
  }

  return { open };
})();

function downloadBlob(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

function exportJSON() {
  const job = state.currentJob;
  if (!job) return toast.warn('没有可导出的任务');
  const summary = {
    id: job.id, kind: job.kind, display_name: job.display_name, status: job.status,
    created_ms: job.created_ms, error: job.error, stats: job.stats,
    face_count: job.face_count, frame_count: job.frame_count,
    frames: (job.frames || []).map(f => ({
      index: f.index, timestamp_ms: f.timestamp_ms,
      annotated_key: f.annotated_key, original_key: f.original_key,
      annotated_url: f.annotated_key ? utils.mediaUrl(f.annotated_key) : null,
      original_url: f.original_key ? utils.mediaUrl(f.original_key) : null,
      faces: (f.faces || []).map(face => ({
        key: face.key, url: utils.mediaUrl(face.key),
        x: face.x, y: face.y, w: face.w, h: face.h, score: face.score,
      })),
    })),
  };
  downloadBlob(JSON.stringify(summary, null, 2), `job-${job.id}.json`, 'application/json');
  toast.success(`已下载 JSON (${summary.frame_count} 帧)`);
}

function exportCSV() {
  const job = state.currentJob;
  if (!job) return toast.warn('没有可导出的任务');
  const rows = [['frame_idx', 'timestamp_ms', 'x', 'y', 'w', 'h', 'score', 'face_url']];
  let n = 0;
  for (const f of job.frames || []) {
    for (const face of f.faces || []) {
      rows.push([f.index, f.timestamp_ms, face.x, face.y, face.w, face.h, face.score, utils.mediaUrl(face.key)]); n++;
    }
  }
  const csv = rows.map(r => r.map(v => {
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }).join(',')).join('\n');
  downloadBlob(csv, `faces-${job.id}.csv`, 'text/csv');
  toast.success(`已下载 CSV (${n} 行)`);
}

async function exportZip() {
  const job = state.currentJob;
  if (!job) return toast.warn('没有可导出的任务');
  if (job.kind === 'image' && !job.original_key) {
    return toast.warn('该任务没有可下载的内容');
  }
  // 直接通过 <a> 触发后端 zip 端点,Content-Disposition 强制下载。
  const url = '/api/jobs/' + encodeURIComponent(job.id) + '/download.zip';
  const a = document.createElement('a');
  a.href = url;
  a.download = ''; // 让浏览器尊重 Content-Disposition
  document.body.appendChild(a);
  a.click();
  setTimeout(() => a.remove(), 100);
  toast.success(`已请求下载 ${job.kind} 结果包(annotated + faces + manifest.json)`);
}

async function cancelCurrent() {
  if (!state.currentJobId) return;
  try { await api.cancelJob(state.currentJobId); toast.info('已发送停止信号'); }
  catch (e) { toast.error('取消失败: ' + e.message); }
}

async function deleteCurrent() {
  if (!state.currentJobId) return;
  const id = state.currentJobId;
  const name = state.currentJob ? (state.currentJob.display_name || id) : id;
  if (!await confirmModal.open('删除任务', `确认删除 "${name}"?此操作不可恢复。`)) return;
  try {
    await api.deleteJob(id);
    state.deletedIds.add(id);
    if (state.currentJobId === id) preview.close();
    sidebar.render();
    toast.success('任务已删除');
  } catch (e) { toast.error('删除失败: ' + e.message); }
}

async function retryCurrent() {
  if (!state.currentJobId) return;
  const id = state.currentJobId;
  if (!await confirmModal.open('重跑任务', '将用原 URL/输入重新创建一个任务。继续?')) return;
  try {
    const r = await api.retryJob(id);
    toast.success('已重跑,新任务 #' + (r.job_id || '').slice(0, 8));
    if (r.job_id) {
      const job = await api.getJob(r.job_id);
      sidebar.upsertJob(job); preview.open(job.id);
    }
  } catch (e) { toast.error('重跑失败: ' + e.message); }
}

function initKeys() {
  document.addEventListener('keydown', e => {
    const inField = e.target && e.target.matches && e.target.matches('input, textarea, [contenteditable]');
    if (e.key === 'Escape') {
      let closed = false;
      utils.$$('.modal').forEach(m => { if (!m.classList.contains('hidden')) { m.classList.add('hidden'); closed = true; } });
      if (closed) { e.preventDefault(); return; }
      if (batch.isActive()) { batch.exit(); e.preventDefault(); return; }
      if (state.currentJobId) { preview.close(); e.preventDefault(); return; }
    }
    if (inField) return;
    if ((e.ctrlKey || e.metaKey) && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault();
      const s = utils.$('#tb-search'); if (s) { s.focus(); s.select(); }
      return;
    }
    if (e.key === 'n' || e.key === 'N') { upload.openModal(); e.preventDefault(); return; }
    if (e.key === 'ArrowDown') { sidebar.nextItem(1); e.preventDefault(); return; }
    if (e.key === 'ArrowUp')   { sidebar.nextItem(-1); e.preventDefault(); return; }
    // Home / End / PageUp / PageDown:整段跳
    if (e.key === 'Home') {
      sidebar.gotoOffset(0); e.preventDefault(); return;
    }
    if (e.key === 'End') {
      sidebar.gotoOffset(-1); e.preventDefault(); return;
    }
    if (e.key === 'PageDown' || e.key === 'PageUp') {
      const step = Math.max(1, Math.floor(
        (state.listScrollEl ? state.listScrollEl.clientHeight : 400) / ROW_H) - 1);
      const list = sidebar.filtered();
      const cur = list.findIndex(j => j.id === state.currentJobId);
      if (cur < 0) { sidebar.gotoOffset(0); }
      else if (e.key === 'PageDown') sidebar.gotoOffset(cur + step);
      else sidebar.gotoOffset(cur - step);
      e.preventDefault(); return;
    }
    if (e.key === 'a' || e.key === 'A') { preview.toggleAnno(); e.preventDefault(); return; }
    if (e.key === 'Delete' && state.currentJobId) { deleteCurrent(); e.preventDefault(); return; }
    if (e.key === '?') { utils.$('#modal-help').classList.remove('hidden'); e.preventDefault(); return; }
    if (e.key === 'b' || e.key === 'B') { if (!batch.isActive()) batch.enter(); else batch.exit(); e.preventDefault(); return; }
  });
}

async function init() {
  theme.init();
  sidebar.init(); upload.init(); batch.init(); confirmModal.init(); initKeys();
  // 平台 KPI 实时拉取
  if (typeof kpi !== 'undefined' && kpi && typeof kpi.init === 'function') kpi.init();
  // 视频双画面播放器:共享控制 + 拖动分隔条 + 人脸筛选 + 人脸 lightbox
  if (typeof preview.initSharedControls === 'function') preview.initSharedControls();
  if (typeof preview.initDivider === 'function') preview.initDivider();
  if (typeof preview.initFaceFilters === 'function') preview.initFaceFilters();
  if (typeof lightbox.init === 'function') lightbox.init();
  // 顶栏主题快切按钮(在 theme.init() 之后,确保按钮已存在)
  const tbTheme = utils.$('#tb-theme');
  if (tbTheme) tbTheme.addEventListener('click', theme.cycle);
  utils.$('#pv-close').addEventListener('click', preview.close);
  utils.$('#pv-cancel').addEventListener('click', cancelCurrent);
  utils.$('#pv-toggle-anno').addEventListener('click', preview.toggleAnno);
  utils.$('#pv-retry').addEventListener('click', retryCurrent);
  utils.$('#pv-delete').addEventListener('click', deleteCurrent);
  utils.$('#pv-export-json').addEventListener('click', exportJSON);
  utils.$('#pv-export-csv').addEventListener('click', exportCSV);
  utils.$('#pv-export-zip').addEventListener('click', exportZip);
  // URL hash 深链:刷新 / 分享链接可定位到指定 job
  hashRouter.bind();
  try {
    const jobs = await api.listJobs();
    sidebar.setJobs(jobs);
    hashRouter.maybeOpen();
  } catch (e) { toast.error('加载任务列表失败: ' + e.message); }
  api.getConfig().then(c => { state.config = c; }).catch(() => {});
}

/**
 * URL hash 路由:
 *  - 打开 job: #/job/<id>    →  自动 preview.open(id)
 *  - 关闭 job: #/              →  preview.close()
 *  - 浏览器前进/后退自动同步
 *  - 任务打开/关闭时主动更新 hash(replaceState,避免污染历史)
 */
const hashRouter = (() => {
  function parse() {
    const h = location.hash.replace(/^#\/?/, '');
    const m = h.match(/^job\/(.+)$/);
    return m ? { kind: 'job', id: m[1] } : null;
  }
  function setJob(id, replace = true) {
    const want = '#/job/' + encodeURIComponent(id);
    if (location.hash === want) return;
    if (replace) history.replaceState(null, '', want);
    else location.hash = want;
  }
  function clear(replace = true) {
    if (!location.hash) return;
    if (replace) history.replaceState(null, '', location.pathname + location.search);
    else location.hash = '';
  }
  function bind() {
    window.addEventListener('hashchange', () => {
      const r = parse();
      if (r && r.kind === 'job' && r.id !== state.currentJobId) preview.open(r.id);
      else if (!r && state.currentJobId) preview.close();
    });
  }
  function maybeOpen() {
    const r = parse();
    if (r && r.kind === 'job' && r.id) {
      // 等待 listJobs 完成 → preview 自动打开
      preview.open(r.id);
    }
  }
  return { bind, setJob, clear, parse };
})();

document.addEventListener('DOMContentLoaded', init);
window.__rsface = { state, api, sidebar, preview, sse, batch, theme, toast, dashboard, lightbox, hashRouter };
