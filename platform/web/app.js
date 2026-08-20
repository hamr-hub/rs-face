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
  function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
  function throttleRaf(fn) { let s = false, la = null; return (...a) => { la = a; if (s) return; s = true; requestAnimationFrame(() => { s = false; fn(...la); }); }; }
  /** Bug 1/4: URL-encode media keys so 'local://jobs/...' works in <video>/<img>. */
  function mediaUrl(key) {
    if (!key) return '';
    if (/^(https?:|data:|blob:)/.test(key)) return key;
    return '/media/' + encodeURIComponent(key);
  }
  return { $, $$, toast: legacyToast, fmtTime, fmtAbsTime, escapeHtml, debounce, throttleRaf, mediaUrl };
})();

const state = {
  jobs: [], filter: 'all', search: '', searchMode: 'plain', searchRange: 'all',
  currentJobId: null, currentJob: null,
  annoVisible: true, eventSource: null, faceCardObserver: null,
  listScrollEl: null, listVpEl: null, listSpacerEl: null,
  itemHeight: 60, itemGap: 6,
  batchMode: false, selectedIds: new Set(),
  themePref: 'auto',
  deletedIds: new Set(),
  config: null,
};

const theme = (() => {
  const KEY = 'rsface.theme';
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
  }
  function init() {
    state.themePref = readPref();
    applyPref(state.themePref);
    utils.$$('#set-theme .seg-btn').forEach(b => b.addEventListener('click', () => {
      applyPref(b.dataset.theme); writePref(b.dataset.theme);
    }));
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => {
        if (state.themePref === 'auto') applyPref('auto');
      });
    }
  }
  return { init, applyPref };
})();

const toast = (() => {
  const host = () => utils.$('#toast-host');
  function show(msg, kind = 'info', ms = 3500) {
    const h = host(); if (!h) return;
    const el = document.createElement('div');
    el.className = 'toast-item ' + kind;
    el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
    const msgEl = document.createElement('span'); msgEl.className = 'toast-msg'; msgEl.textContent = msg;
    const closeEl = document.createElement('button'); closeEl.className = 'toast-x'; closeEl.setAttribute('aria-label', '关闭'); closeEl.textContent = '×';
    el.appendChild(msgEl); el.appendChild(closeEl);
    h.appendChild(el);
    const close = () => { el.classList.add('leaving'); setTimeout(() => el.remove(), 200); };
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
    const thumbSrc = j.original_key ? utils.mediaUrl(j.original_key) : null;
    let thumbHtml;
    if (j.status === 'running' || j.status === 'queued') thumbHtml = `<div class="sb-thumb"><div style="opacity:.6">⏳</div></div>`;
    else if (thumbSrc) thumbHtml = `<div class="sb-thumb"><img data-src="${utils.escapeHtml(thumbSrc)}" alt=""></div>`;
    else thumbHtml = `<div class="sb-thumb"><div>·</div></div>`;
    el.innerHTML = `
      <div class="sb-check" aria-hidden="true"><input type="checkbox" ${state.selectedIds.has(j.id) ? 'checked' : ''} tabindex="-1"></div>
      ${thumbHtml}
      <div class="sb-info">
        <div class="sb-name" title="${utils.escapeHtml(j.display_name || j.id)}">${utils.escapeHtml(j.display_name || j.id)}</div>
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
  }

  function nextItem(dir) {
    const list = filtered(); if (!list.length) return;
    let i = list.findIndex(j => j.id === state.currentJobId);
    if (i < 0) i = 0; else i = Math.max(0, Math.min(list.length - 1, i + dir));
    preview.open(list[i].id);
  }

  return { init, setJobs, upsertJob, render, setActive, nextItem, filtered };
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
  }

  function render(job) {
    utils.$('#pv-title').textContent = job.display_name || job.id;
    utils.$('#pv-id').textContent = '#' + (job.id || '').slice(0, 8);
    utils.$('#pv-kind').textContent = job.kind || '?';
    const st = job.stats || {};
    utils.$('#pv-frames').textContent = `${st.frames_processed || 0} / ${job.frame_count || st.frames_processed || 0} 帧`;
    utils.$('#pv-faces').textContent = `${job.face_count || 0} 张脸`;
    utils.$('#pv-time').textContent = utils.fmtAbsTime(job.created_ms);
    const status = job.status || 'queued';
    const statusEl = utils.$('#pv-status'); statusEl.textContent = status; statusEl.className = 'pv-status ' + status;
    utils.$('#pv-dot').className = 'pv-dot ' + status;
    utils.$('#pv-cancel').classList.toggle('hidden', !(status === 'running' || status === 'queued'));
    // Bug 2:error/cancelled 都显示原图 + 顶部红色 banner
    const banner = utils.$('#pv-error-banner');
    if (status === 'error' || status === 'cancelled') {
      const label = status === 'cancelled' ? '任务已取消' : '识别失败';
      const msg = job.error ? `${label} · ${job.error}` : label;
      if (banner) { banner.textContent = msg; banner.classList.remove('hidden'); }
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
    if (job.kind === 'image') renderImage(job);
    else if (job.kind === 'video') renderVideo(job);
    else renderStream(job);
    renderProgress(job); renderFaceGrid(job);
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
      img.src = '/media/' + src;
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
    // 标注视频:job.annotated_key 是后端合成的 mp4;没有就 fall back 到首帧 poster
    if (job.annotated_key && anno) anno.src = utils.mediaUrl(job.annotated_key);
    if (anno) bindSync(orig, anno);
    orig.onloadedmetadata = () => {
      hideHint();
      const first = (job.frames || []).find(f => f.annotated_key);
      if (first && anno && !anno.src) anno.poster = utils.mediaUrl(first.annotated_key);
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
        img.src = '/media/' + job.original_key;
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
    const faces = [];
    for (const f of job.frames || []) {
      for (const face of (f.faces || [])) {
        faces.push({ ts: f.timestamp_ms, key: face.key, score: face.score, frame: f, x: face.x, y: face.y, w: face.w, h: face.h });
      }
    }
    faces.sort((a, b) => a.ts - b.ts);
    const clusters = clusterFaces(faces);
    utils.$('#pv-faces-count').textContent = clusters.length ? `${faces.length} 张 · ${clusters.length} 组` : '—';
    if (!clusters.length) { grid.innerHTML = '<div class="empty">尚未出现人脸</div>'; return; }
    grid.innerHTML = '';
    for (const cl of clusters) {
      const rep = cl.rep;
      const card = document.createElement('div');
      card.className = 'face-card';
      card.innerHTML = `
        <img loading="lazy" decoding="async" data-src="${utils.escapeHtml(utils.mediaUrl(rep.key))}" alt="face">
        <div class="fc-time">⏱ ${utils.fmtTime(rep.ts)}</div>
        <div class="fc-score">score ${rep.score.toFixed(2)}</div>
        ${cl.members.length > 1 ? `<div class="fc-badge">×${cl.members.length}</div>` : ''}`;
      card.addEventListener('click', () => {
        if (state.currentJob && state.currentJob.kind === 'video') {
          const vid = utils.$('#pv-vid');
          if (vid && rep.frame) { vid.currentTime = rep.frame.timestamp_ms / 1000; vid.play().catch(() => {}); }
        } else if (state.currentJob && state.currentJob.kind === 'video') { const o = utils.$('#pv-orig'); if (o && rep.frame) { o.currentTime = rep.frame.timestamp_ms / 1000; o.play().catch(() => {}); }
        }
      });
      grid.appendChild(card);
    }
    grid.querySelectorAll('img[data-src]').forEach(img => state.faceCardObserver.observe(img));
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
    const btn = utils.$('#pv-toggle-anno');
    btn.textContent = state.annoVisible ? '标注 ●' : '标注 〇';
    btn.setAttribute('aria-pressed', state.annoVisible ? 'true' : 'false');
    if (state.currentJob) render(state.currentJob);
  }

  return { open, close, render, toggleAnno, showEmpty, showDetail };
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
  function attach(jobId) {
    detach();
    if (!('EventSource' in window)) return;
    const es = new EventSource('/api/jobs/' + encodeURIComponent(jobId) + '/events');
    state.eventSource = es;
    es.onmessage = (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === 'frame') scheduleRefresh();
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
  return { attach, detach };
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
      annotated_url: f.annotated_key ? '/media/' + f.annotated_key : null,
      original_url: f.original_key ? '/media/' + f.original_key : null,
      faces: (f.faces || []).map(face => ({
        key: face.key, url: '/media/' + face.key,
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
    if (e.key === 'a' || e.key === 'A') { preview.toggleAnno(); e.preventDefault(); return; }
    if (e.key === 'Delete' && state.currentJobId) { deleteCurrent(); e.preventDefault(); return; }
    if (e.key === '?') { utils.$('#modal-help').classList.remove('hidden'); e.preventDefault(); return; }
    if (e.key === 'b' || e.key === 'B') { if (!batch.isActive()) batch.enter(); else batch.exit(); e.preventDefault(); return; }
  });
}

async function init() {
  theme.init();
  sidebar.init(); upload.init(); batch.init(); confirmModal.init(); initKeys();
  // 视频双画面播放器:共享控制 + 拖动分隔条
  if (typeof preview.initSharedControls === 'function') preview.initSharedControls();
  if (typeof preview.initDivider === 'function') preview.initDivider();
  utils.$('#pv-close').addEventListener('click', preview.close);
  utils.$('#pv-cancel').addEventListener('click', cancelCurrent);
  utils.$('#pv-toggle-anno').addEventListener('click', preview.toggleAnno);
  utils.$('#pv-retry').addEventListener('click', retryCurrent);
  utils.$('#pv-delete').addEventListener('click', deleteCurrent);
  utils.$('#pv-export-json').addEventListener('click', exportJSON);
  utils.$('#pv-export-csv').addEventListener('click', exportCSV);
  try {
    const jobs = await api.listJobs();
    sidebar.setJobs(jobs);
  } catch (e) { toast.error('加载任务列表失败: ' + e.message); }
  api.getConfig().then(c => { state.config = c; }).catch(() => {});
}

document.addEventListener('DOMContentLoaded', init);
window.__rsface = { state, api, sidebar, preview, sse, batch, theme, toast, dashboard };
