/* rs-face Platform · 零依赖 vanilla JS · 模块:api / utils / sidebar / preview / upload / sse / keys
 * 优化:虚拟滚动(±6 overscan) · 缩略图 IntersectionObserver 懒加载 · SSE 帧事件 throttleRaf 批渲染
 *     · <img> 全部 loading="lazy" decoding="async" · 无 setInterval 长轮询 */
'use strict';

const api = {
  listJobs:    () => fetch('/api/jobs').then(r => r.ok ? r.json().then(j => j.jobs || []) : Promise.reject(new Error(r.status))),
  getJob:      id => fetch('/api/jobs/' + encodeURIComponent(id)).then(r => r.ok ? r.json() : Promise.reject(new Error(r.status))),
  cancelJob:   id => fetch('/api/jobs/' + encodeURIComponent(id) + '/cancel', { method: 'POST' }),
  postImage:   file => { const fd = new FormData(); fd.append('file', file); return fetch('/api/jobs/image', { method: 'POST', body: fd }).then(r => r.json()); },
  postVideo:   file => { const fd = new FormData(); fd.append('file', file); return fetch('/api/jobs/video', { method: 'POST', body: fd }).then(r => r.json()); },
  postStream:  url => fetch('/api/jobs/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url }) }).then(r => r.json()),
  getConfig:   () => fetch('/api/config').then(r => r.ok ? r.json() : null).catch(() => null),
};

const utils = (() => {
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  function toast(msg, isError) {
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
  /**
   * 把 S3/local key 拼成可用的媒体 URL。
   * Bug 1/4 修复:必须 encodeURIComponent,否则 `local://jobs/...` 中的
   * `://` 会被浏览器解析成 scheme,导致 <video>/<img> 拒绝加载。
   * 如果 key 已经是 http(s)/data/blob 形式则原样返回。
   */
  function mediaUrl(key) {
    if (!key) return '';
    if (/^(https?:|data:|blob:)/.test(key)) return key;
    return '/media/' + encodeURIComponent(key);
  }
  return { $, $$, toast, fmtTime, fmtAbsTime, escapeHtml, debounce, throttleRaf, mediaUrl };
})();

const state = {
  jobs: [], filter: 'all', search: '', currentJobId: null, currentJob: null,
  annoVisible: true, eventSource: null, faceCardObserver: null,
  listScrollEl: null, listVpEl: null, listSpacerEl: null,
  itemHeight: 60, itemGap: 6,
};

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
      utils.$$('#sb-filters .sb-filter').forEach(x => x.classList.toggle('active', x === b));
      render();
    }));
    const search = utils.$('#tb-search');
    const onSearch = utils.debounce(() => { state.search = (search.value || '').toLowerCase().trim(); render(); }, 120);
    search.addEventListener('input', onSearch);
  }

  function filtered() {
    let jobs = state.jobs || [];
    if (state.filter === 'running') jobs = jobs.filter(j => j.status === 'running' || j.status === 'queued');
    else if (state.filter === 'done') jobs = jobs.filter(j => j.status === 'done');
    else if (state.filter === 'error') jobs = jobs.filter(j => j.status === 'error' || j.status === 'cancelled');
    if (state.search) {
      const q = state.search;
      jobs = jobs.filter(j => (j.display_name || '').toLowerCase().includes(q) || (j.id || '').toLowerCase().includes(q));
    }
    return jobs;
  }

  function setJobs(jobs) { state.jobs = jobs; render(); }

  function upsertJob(j) {
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
    if (j.id === state.currentJobId) el.classList.add('active');
    el.addEventListener('click', () => preview.open(j.id));
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
    const img = el.querySelector('img[data-src]');
    if (img) state.faceCardObserver.observe(img);
  }

  function setActive(id) { utils.$$('.sb-item').forEach(e => e.classList.toggle('active', e.id === 'sb-' + id)); }

  function nextItem(dir) {
    const list = filtered(); if (!list.length) return;
    let i = list.findIndex(j => j.id === state.currentJobId);
    if (i < 0) i = 0; else i = Math.max(0, Math.min(list.length - 1, i + dir));
    preview.open(list[i].id);
  }

  return { init, setJobs, upsertJob, render, setActive, nextItem };
})();

const preview = (() => {
  let lastFrameIdx = -1, _lastAnnoFrameIdx = -1;
  // 双画面同步状态
  let _syncLocked = false;

  const showEmpty = () => { utils.$('#pv-empty').classList.remove('hidden'); utils.$('#pv-detail').classList.add('hidden'); };
  const showDetail = () => { utils.$('#pv-empty').classList.add('hidden'); utils.$('#pv-detail').classList.remove('hidden'); };
  const showHint = t => { const h = utils.$('#pv-stage-hint'); h.textContent = t; h.classList.remove('hidden'); };
  const hideHint = () => utils.$('#pv-stage-hint').classList.add('hidden');

  function resetMedia() {
    const img = utils.$('#pv-img');
    if (img) { img.removeAttribute('src'); img.classList.add('hidden'); }
    const dbl = utils.$('#pv-double');
    if (dbl) dbl.classList.add('hidden');
    const sc = utils.$('#pv-shared-controls');
    if (sc) sc.classList.add('hidden');
    const o = utils.$('#pv-orig'), a = utils.$('#pv-anno');
    if (o) { o.removeAttribute('src'); o.pause(); }
    if (a) { a.removeAttribute('src'); a.pause(); }
    const err = utils.$('#pv-error-banner');
    if (err) { err.classList.add('hidden'); err.textContent = ''; }
    clearOverlay();
  }

  async function open(jobId) {
    state.currentJobId = jobId; lastFrameIdx = -1; _lastAnnoFrameIdx = -1;
    sidebar.setActive(jobId); showDetail(); showHint('加载中…'); resetMedia(); sse.detach();
    try {
      const job = await api.getJob(jobId);
      state.currentJob = job; render(job);
      if (job.status === 'running' || job.status === 'queued') sse.attach(jobId);
    } catch (e) { utils.toast('加载任务失败: ' + e.message, true); }
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
    // 错误 banner(error / cancelled 状态显示)
    const errEl = utils.$('#pv-error-banner');
    if (job.error && (status === 'error' || status === 'cancelled')) {
      errEl.textContent = (status === 'cancelled' ? '任务已取消' : '识别失败') + ' · ' + job.error;
      errEl.classList.remove('hidden');
    } else { errEl.classList.add('hidden'); errEl.textContent = ''; }
    if (job.kind === 'image') renderImage(job);
    else if (job.kind === 'video') renderVideo(job);
    else renderStream(job);
    renderProgress(job); renderFaceGrid(job);
  }

  function renderImage(job) {
    const img = utils.$('#pv-img');
    const dbl = utils.$('#pv-double'); if (dbl) dbl.classList.add('hidden');
    const sc = utils.$('#pv-shared-controls'); if (sc) sc.classList.add('hidden');
    img.classList.remove('hidden');
    const frames = job.frames || [];
    const first = frames.find(f => f.original_key) || frames[0];
    // 优先用 frame.original_key,没有时降级到 job.original_key(error 时只有这个)
    const src = (first && first.original_key) || job.original_key;
    if (src) {
      img.onload = () => { hideHint(); if (first) drawOverlay(job, first, img); };
      img.onerror = () => { showHint('原图加载失败'); clearOverlay(); };
      img.src = utils.mediaUrl(src);
    } else { img.classList.add('hidden'); showHint('无原图'); clearOverlay(); }
  }

  /**
   * 双画面视频播放器。
   * 左边 pv-orig 播原视频(job.original_key);
   * 右边 pv-anno 播标注视频(由 jobs.rs 在 run_job 末尾合成,或流模式下
   *   用最新帧的 annotated_key 通过 SSE 推送);
   * 两侧用 bindSync 互锁:play/pause/seeked/ratechange/ended 全镜像。
   * 顶部共享进度条(shared progress)基于 pv-orig.currentTime / duration。
   */
  function renderVideo(job) {
    const img = utils.$('#pv-img'); if (img) img.classList.add('hidden');
    const dbl = utils.$('#pv-double');
    const sc = utils.$('#pv-shared-controls');
    if (dbl) dbl.classList.remove('hidden');
    if (sc) sc.classList.remove('hidden');
    const o = utils.$('#pv-orig'), a = utils.$('#pv-anno');
    if (!o || !a) return;
    const origKey = job.original_key;
    const annoKey = job.annotated_key; // 视频任务的合成标注 mp4(若后端有)
    if (origKey) {
      const newSrc = utils.mediaUrl(origKey);
      if (o.src !== newSrc) o.src = newSrc;
    } else {
      o.removeAttribute('src');
    }
    if (annoKey) {
      const newAnno = utils.mediaUrl(annoKey);
      if (a.src !== newAnno) a.src = newAnno;
    } else {
      // 没有合成 mp4 时,把最新一帧的 annotated_key 当 poster 显示
      const f0 = (job.frames || []).slice().reverse().find(f => f.annotated_key);
      if (f0) a.poster = utils.mediaUrl(f0.annotated_key);
    }
    bindSync(o, a);
    o.onloadedmetadata = () => {
      hideHint();
      const f0 = (job.frames || []).find(f => f.annotated_key);
      if (f0) drawOverlayOnAnno(job, f0);
      refreshSharedProgress();
    };
    a.onloadedmetadata = () => { hideHint(); refreshSharedProgress(); };
    o.ontimeupdate = () => { syncVideoOverlayDual(job); refreshSharedProgress(); };
    a.ontimeupdate = () => { syncOrigByAnnoTime(job); refreshSharedProgress(); };
  }

  function syncVideoOverlayDual(job) {
    const o = utils.$('#pv-orig'); if (!o) return;
    const frames = job.frames || []; if (!frames.length) return;
    const cur = o.currentTime * 1000; let best = null;
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
    const a = utils.$('#pv-anno'), o = utils.$('#pv-orig');
    if (!a || !o) return;
    if (Math.abs(o.currentTime - a.currentTime) > 0.4) o.currentTime = a.currentTime;
  }
  function drawOverlayOnAnno(job, frame) {
    if (!state.annoVisible) return clearOverlay();
    const a = utils.$('#pv-anno'), c = utils.$('#pv-overlay'), stage = utils.$('#pv-stage');
    if (!a || !c || !stage) return;
    const rect = stage.getBoundingClientRect(), m = a.getBoundingClientRect();
    c.style.left = (m.left - rect.left) + 'px';
    c.style.top = (m.top - rect.top) + 'px';
    c.width = m.width; c.height = m.height;
    const natW = a.videoWidth || (frame.faces[0] ? frame.faces[0].w * 4 : 640);
    const natH = a.videoHeight || 360;
    drawBoxes(c, frame, natW, natH);
  }

  /**
   * 双 video 互锁:一边 play/pause/seeked/ratechange/ended,
   * 另一边立即镜像,误差 < 100ms。_syncLocked 防反弹。
   */
  function bindSync(a, b) {
    if (a._syncBoundTo === b) return;
    a._syncBoundTo = b; b._syncBoundTo = a;
    const mirror = (src, dst) => () => {
      if (_syncLocked) return;
      _syncLocked = true;
      try {
        if (src.playbackRate && dst.playbackRate !== src.playbackRate) dst.playbackRate = src.playbackRate;
        if (Math.abs((dst.currentTime || 0) - (src.currentTime || 0)) > 0.12) dst.currentTime = src.currentTime;
        if (!src.paused && dst.paused) dst.play().catch(() => {});
        if (src.paused && !dst.paused) dst.pause();
      } finally {
        requestAnimationFrame(() => { _syncLocked = false; });
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

  function renderStream(job) {
    const img = utils.$('#pv-img');
    const dbl = utils.$('#pv-double');
    const sc = utils.$('#pv-shared-controls');
    if (dbl) dbl.classList.remove('hidden');
    if (sc) sc.classList.remove('hidden');
    img.classList.add('hidden');
    const o = utils.$('#pv-orig'), a = utils.$('#pv-anno');
    const frames = job.frames || [];
    const last = [...frames].reverse().find(f => f.original_key) || frames[frames.length - 1];
    if (!last) {
      // 没有 frame 时降级到 job.original_key
      if (job.original_key) {
        if (o) o.src = utils.mediaUrl(job.original_key);
        if (o) o.onload = () => hideHint();
        if (o) o.onerror = () => showHint('原图加载失败');
      } else { showHint('等待原始帧…'); clearOverlay(); }
      return;
    }
    if (last.index === lastFrameIdx) return;
    lastFrameIdx = last.index;
    if (o) { o.src = utils.mediaUrl(last.original_key) + (last.original_key.includes('?') ? '&' : '?') + 'v=' + last.index; o.onload = () => { hideHint(); drawOverlay(job, last, o); }; }
    if (a && last.annotated_key) { a.src = utils.mediaUrl(last.annotated_key) + (last.annotated_key.includes('?') ? '&' : '?') + 'v=' + last.index; }
  }

  function clearOverlay() {
    const c = utils.$('#pv-overlay'); c.width = 0; c.height = 0;
    const ctx = c.getContext('2d'); if (ctx) ctx.clearRect(0, 0, c.width, c.height);
  }

  function drawOverlay(job, frame, mediaEl) {
    if (!state.annoVisible) return clearOverlay();
    const c = utils.$('#pv-overlay'), stage = utils.$('#pv-stage');
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
        if (!img.classList.contains('hidden')) {
          const f = (job.frames || []).find(fr => fr.original_key && fr.index === lastFrameIdx) || (job.frames || []).find(fr => fr.original_key);
          if (f) drawOverlay(job, f, img);
        }
      } else if (job.kind === 'video') {
        if (!vid.classList.contains('hidden')) {
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
    const marks = utils.$('#pv-prog-marks'); marks.innerHTML = '';
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
    if (state.currentJob) render(state.currentJob);
  }

  function initSharedControls() {
    const play = utils.$('#pv-shared-play');
    const track = utils.$('#pv-shared-track');
    const rate = utils.$('#pv-shared-rate');
    if (play) play.addEventListener('click', () => {
      const o = utils.$('#pv-orig'); if (!o) return;
      if (o.paused) o.play().catch(() => {}); else o.pause();
      play.textContent = (o.paused ? '▶' : '⏸');
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
    // 键盘 Space 控制 play/pause
    document.addEventListener('keydown', (e) => {
      if (e.target && e.target.matches && e.target.matches('input, textarea, [contenteditable]')) return;
      if (e.key === ' ' && state.currentJob && (state.currentJob.kind === 'video' || state.currentJob.kind === 'stream')) {
        const o = utils.$('#pv-orig'); if (!o) return;
        e.preventDefault();
        if (o.paused) o.play().catch(() => {}); else o.pause();
      }
    });
    // 同步播放/暂停按钮图标
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

  return { open, close, render, toggleAnno, showEmpty, showDetail, initSharedControls, initDivider };
})();

const upload = (() => {
  function init() {
    utils.$('#tb-new').addEventListener('click', openModal);
    utils.$('#tb-settings').addEventListener('click', openSettings);
    utils.$$('.new-type').forEach(b => b.addEventListener('click', () => selectType(b.dataset.type)));
    utils.$$('[data-close]').forEach(el => el.addEventListener('click', closeAllModals));
    setupDropzone('#dz-image', '#file-image', files => {
      if (files.length > 1) for (const f of files) submitImage(f); else submitImage(files[0]);
      closeAllModals();
    });
    setupDropzone('#dz-video', '#file-video', files => { submitVideo(files[0]); closeAllModals(); });
    utils.$('#video-url-go').addEventListener('click', () => {
      const url = utils.$('#video-url').value.trim();
      if (!url) return utils.toast('请输入视频 URL', true);
      submitStream(url); closeAllModals();
    });
    utils.$('#stream-go').addEventListener('click', () => {
      const url = utils.$('#stream-url').value.trim();
      if (!url) return utils.toast('请输入流地址', true);
      submitStream(url); closeAllModals();
    });
    setupGlobalDrop(); setupGlobalPaste();
  }

  const openModal = () => { utils.$('#modal-new').classList.remove('hidden'); selectType('image'); };
  const openSettings = () => { utils.$('#modal-settings').classList.remove('hidden'); loadSettings(); };
  const closeAllModals = () => utils.$$('.modal').forEach(m => m.classList.add('hidden'));
  function selectType(t) {
    utils.$$('.new-type').forEach(b => b.classList.toggle('active', b.dataset.type === t));
    utils.$$('.new-pane').forEach(p => p.classList.add('hidden'));
    const pane = utils.$('#pane-' + t); if (pane) pane.classList.remove('hidden');
  }
  async function loadSettings() {
    const cfg = await api.getConfig();
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
      else utils.toast('不支持的文件类型: ' + t, true);
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
      utils.toast(`上传 ${file.name}…`);
      const data = await api.postImage(file);
      if (data.error) return utils.toast('上传失败: ' + data.error, true);
      utils.toast('已提交 #' + (data.job_id || '').slice(0, 8));
      try {
        const job = await api.getJob(data.job_id);
        sidebar.upsertJob(job); preview.open(job.id);
      } catch {}
    } catch (e) { utils.toast('上传失败: ' + e.message, true); }
  }
  async function submitVideo(file) {
    try {
      utils.toast(`上传 ${file.name}…`);
      const data = await api.postVideo(file);
      if (data.error) return utils.toast('上传失败: ' + data.error, true);
      utils.toast('已提交 #' + (data.job_id || '').slice(0, 8));
      const job = await api.getJob(data.job_id);
      sidebar.upsertJob(job); preview.open(job.id);
    } catch (e) { utils.toast('上传失败: ' + e.message, true); }
  }
  async function submitStream(url) {
    try {
      const data = await api.postStream(url);
      if (data.error) return utils.toast('启动失败: ' + data.error, true);
      utils.toast('已启动流 #' + (data.job_id || '').slice(0, 8));
      const job = await api.getJob(data.job_id);
      sidebar.upsertJob(job); preview.open(job.id);
    } catch (e) { utils.toast('启动失败: ' + e.message, true); }
  }
  return { init, openModal, openSettings };
})();

const sse = (() => {
  const scheduleRefresh = utils.throttleRaf(async () => {
    if (!state.currentJobId) return;
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
        utils.toast(
          msg.type === 'done' ? '任务完成' : msg.type === 'cancelled' ? '已停止' : '任务出错: ' + (msg.message || ''),
          msg.type === 'error'
        );
      }
    };
    es.onerror = () => { /* browser auto-retry */ };
  }
  function detach() { if (state.eventSource) { state.eventSource.close(); state.eventSource = null; } }
  return { attach, detach };
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
  if (!job) return utils.toast('没有可导出的任务', true);
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
        key: face.key, url: '/media/' + face.key,
        x: face.x, y: face.y, w: face.w, h: face.h, score: face.score,
      })),
    })),
  };
  downloadBlob(JSON.stringify(summary, null, 2), `job-${job.id}.json`, 'application/json');
  utils.toast(`已下载 JSON (${summary.frame_count} 帧)`);
}

function exportCSV() {
  const job = state.currentJob;
  if (!job) return utils.toast('没有可导出的任务', true);
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
  utils.toast(`已下载 CSV (${n} 行)`);
}

async function cancelCurrent() {
  if (!state.currentJobId) return;
  try { await api.cancelJob(state.currentJobId); utils.toast('已发送停止信号'); }
  catch (e) { utils.toast('取消失败: ' + e.message, true); }
}

function initKeys() {
  document.addEventListener('keydown', e => {
    const inField = e.target && e.target.matches && e.target.matches('input, textarea, [contenteditable]');
    if (e.key === 'Escape') {
      let closed = false;
      utils.$$('.modal').forEach(m => { if (!m.classList.contains('hidden')) { m.classList.add('hidden'); closed = true; } });
      if (closed) { e.preventDefault(); return; }
      if (state.currentJobId) { preview.close(); e.preventDefault(); return; }
    }
    if (inField) return;
    if (e.key === 'n' || e.key === 'N') { upload.openModal(); e.preventDefault(); return; }
    if (e.key === 'ArrowDown') { sidebar.nextItem(1); e.preventDefault(); return; }
    if (e.key === 'ArrowUp')   { sidebar.nextItem(-1); e.preventDefault(); return; }
    if (e.key === 'a' || e.key === 'A') { preview.toggleAnno(); e.preventDefault(); return; }
  });
}

async function init() {
  sidebar.init(); upload.init(); initKeys();
  utils.$('#pv-close').addEventListener('click', preview.close);
  utils.$('#pv-cancel').addEventListener('click', cancelCurrent);
  utils.$('#pv-toggle-anno').addEventListener('click', preview.toggleAnno);
  utils.$('#pv-export-json').addEventListener('click', exportJSON);
  utils.$('#pv-export-csv').addEventListener('click', exportCSV);
  // Bug 2 修复:error/cancelled 状态显示重跑按钮 → 重新上传同名文件
  const retryBtn = utils.$('#pv-retry');
  if (retryBtn) retryBtn.addEventListener('click', () => {
    const job = state.currentJob; if (!job) return;
    utils.toast('重跑:请重新上传 ' + job.display_name);
    upload.openModal();
  });
  // 双画面视频播放器初始化
  if (typeof preview.initSharedControls === 'function') preview.initSharedControls();
  if (typeof preview.initDivider === 'function') preview.initDivider();
  try {
    const jobs = await api.listJobs();
    sidebar.setJobs(jobs);
  } catch (e) { utils.toast('加载任务列表失败: ' + e.message, true); }
}

document.addEventListener('DOMContentLoaded', init);
window.__rsface = { state, api, sidebar, preview, sse };
