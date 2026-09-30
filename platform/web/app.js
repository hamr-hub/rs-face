/* rs-face Platform · v0.6 主模块
 * 零依赖 vanilla JS(经典 script 共享全局词法作用域):
 *   utils / toast / modal / visibility / upload-queue / dropzone-preview
 *
 * 简洁原则(per platform/DESIGN.md):
 *   - 顶栏 3 元素:logo+title(回家)· search · theme
 *   - 侧栏 1 列(无 tab / 无 algo chip),3 段时间分组(进行 / 今天 / 较早)
 *   - 主区三态:empty / detail — 同一时间只能看到 1 个
 *   - 低频操作收纳进 kebab ⋯ 右键菜单(ctxMenu)
 *   - 全局无 KPI / 仪表板 / 对比 / 活体面板 — 都不在主页
 */
'use strict';

/** POST 响应解析:先 r.ok,再 JSON。错误统一 {error} 形态,纯文本降级。 */
function readApiJson(r) {
  if (!r.ok) {
    return r.json().catch(() => r.text().then(t => ({ error: t || ('HTTP ' + r.status) })))
      .then(j => Promise.reject(new Error(j.error || ('HTTP ' + r.status))));
  }
  return r.json();
}

const api = {
  listJobs() {
    const cached = apiCache.listJobs;
    if (cached && Date.now() - cached.at < 200) return Promise.resolve(cached.value);
    return fetch('/api/jobs').then(r => r.ok ? r.json().then(j => {
      const jobs = j.jobs || [];
      apiCache.listJobs = { at: Date.now(), value: jobs };
      return jobs;
    }) : Promise.reject(new Error(r.status)));
  },
  getJob:      id => fetch('/api/jobs/' + encodeURIComponent(id)).then(r => r.ok ? r.json() : Promise.reject(new Error(r.status))),
  cancelJob:   id => fetch('/api/jobs/' + encodeURIComponent(id) + '/cancel', { method: 'POST' }),
  deleteJob:   id => fetch('/api/jobs/' + encodeURIComponent(id), { method: 'DELETE' }).then(r => r.ok ? r.json() : Promise.reject(new Error(r.status))),
  retryJob:    id => fetch('/api/jobs/' + encodeURIComponent(id) + '/retry', { method: 'POST' }).then(r => r.ok ? r.json() : Promise.reject(new Error(r.status))),
  postImage:   (file, algo) => {
    const fd = new FormData(); fd.append('file', file);
    if (algo) fd.append('algo', algo);
    return fetch('/api/jobs/image', { method: 'POST', body: fd }).then(readApiJson);
  },
  postVideo:   (file, algo) => {
    const fd = new FormData(); fd.append('file', file);
    if (algo) fd.append('algo', algo);
    return fetch('/api/jobs/video', { method: 'POST', body: fd }).then(readApiJson);
  },
  postStream:  (url, algo) => fetch('/api/jobs/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url, algo: algo || undefined }) }).then(readApiJson),
  importVideoUrl: (url, algo) => fetch('/api/import/video-url', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, algo: algo || undefined }),
  }).then(readApiJson),
  getConfig:   () => fetch('/api/config').then(r => r.ok ? r.json() : null).catch(() => null),
};

const apiCache = {
  listJobs: null,
  config:   null,
  bust(key) { if (key) this[key] = null; },
  bustAll() { this.listJobs = null; this.config = null; },
};

const state = {
  jobs: [], search: '',
  currentJobId: null, currentJob: null,
  annoVisible: true, eventSource: null, faceCardObserver: null,
  listScrollEl: null, listVpEl: null, listSpacerEl: null,
  itemHeight: 60, itemGap: 6,
  themePref: 'auto',
  deletedIds: new Set(),
  pinIds: new Set(),
  config: null,
  // 性能采样(FPS / elapsed)
  _fpsSamples: [],
  _fpsLastFrames: 0,
  _fpsLastAt: 0,
};

/** 用户偏好持久化(face 阈值 / 标注显示 / 置顶 / 主题)。 */
const prefStore = (() => {
  const KEYS = {
    annoVisible: 'rsface.annoVisible',
    pinIds: 'rsface.pinIds',
  };
  function read(key, fallback) {
    try { const v = localStorage.getItem(key); return v == null ? fallback : v; } catch { return fallback; }
  }
  function write(key, val) { try { localStorage.setItem(key, String(val)); } catch {} }
  function applyAll() {
    const av = read(KEYS.annoVisible, null);
    if (av != null) state.annoVisible = av === '1';
    const pins = read(KEYS.pinIds, null);
    if (pins) {
      try {
        const arr = JSON.parse(pins);
        if (Array.isArray(arr)) state.pinIds = new Set(arr.filter(x => typeof x === 'string'));
      } catch {}
    }
  }
  return { KEYS, read, write, applyAll };
})();
prefStore.applyAll();

const theme = (() => {
  const KEY = 'rsface.theme';
  function iconFor(pref) {
    if (pref === 'dark') return '◑';
    if (pref === 'light') return '◐';
    return '◐';
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
    const btn = utils.$('#tb-theme');
    if (btn) {
      btn.textContent = iconFor(pref);
      btn.dataset.mode = pref;
      btn.title = pref === 'auto' ? '主题:自动(点击切换)' : `主题:${pref === 'dark' ? '暗' : '亮'}(点击切换)`;
    }
  }
  function cycle() {
    const cur = state.themePref || 'auto';
    const next = cur === 'auto' ? 'dark' : cur === 'dark' ? 'light' : 'auto';
    applyPref(next); writePref(next);
  }
  function init() {
    state.themePref = readPref();
    applyPref(state.themePref);
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

/** 侧栏:单列任务流,3 段时间分组(进行 ≤8 / 今天 24h / 较早),虚拟滚动。 */
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
    const search = utils.$('#tb-search');
    const onSearch = utils.debounce(() => {
      state.search = (search.value || '').trim();
      utils.$('#tb-search-clear').classList.toggle('hidden', !state.search);
      render();
    }, 80);
    search.addEventListener('input', onSearch);
    utils.$('#tb-search-clear').addEventListener('click', () => {
      search.value = ''; state.search = '';
      utils.$('#tb-search-clear').classList.add('hidden');
      render(); search.focus();
    });
  }

  function filtered() {
    let jobs = (state.jobs || []).filter(j => !state.deletedIds.has(j.id));
    if (state.search) {
      const ql = state.search.toLowerCase();
      jobs = jobs.filter(j => (j.display_name || '').toLowerCase().includes(ql) || (j.id || '').toLowerCase().includes(ql));
    }
    if (state.pinIds && state.pinIds.size) {
      jobs = jobs.slice().sort((a, b) => {
        const pa = state.pinIds.has(a.id) ? 0 : 1;
        const pb = state.pinIds.has(b.id) ? 0 : 1;
        return pa - pb;
      });
    }
    return jobs;
  }

  /** 把任务按时间切 3 段:[{kind:'running', jobs:[...]}, {kind:'today', jobs:[...]}, {kind:'earlier', jobs:[...]}] */
  function segmented() {
    const list = filtered();
    const now = Date.now();
    const running = [], today = [], earlier = [];
    for (const j of list) {
      if (j.status === 'running' || j.status === 'queued') running.push(j);
      else if ((now - (j.created_ms || 0)) < 24 * 60 * 60 * 1000) today.push(j);
      else earlier.push(j);
    }
    return [
      ...(running.length ? [{ kind: '进行', jobs: running }] : []),
      ...(today.length   ? [{ kind: '今天', jobs: today }]   : []),
      ...(earlier.length ? [{ kind: '较早', jobs: earlier }] : []),
    ];
  }

  function setJobs(jobs) { state.jobs = jobs; render(); }
  function upsertJob(j) {
    if (state.deletedIds.has(j.id)) return;
    const idx = state.jobs.findIndex(x => x.id === j.id);
    if (idx >= 0) state.jobs[idx] = { ...state.jobs[idx], ...j }; else state.jobs.unshift(j);
    render();
  }

  function runningCount() {
    return (state.jobs || []).filter(j => !state.deletedIds.has(j.id) && (j.status === 'running' || j.status === 'queued')).length;
  }

  function render() {
    const segs = segmented();
    const total = segs.reduce((n, s) => n + s.jobs.length, 0);
    const running = runningCount();
    const footEl = utils.$('#sb-foot');
    if (footEl) footEl.textContent = `共 ${total} 个 · ${running} 运行`;
    const countEl = utils.$('#sb-count');
    if (countEl) countEl.textContent = String(total);
    const sbTotal = utils.$('#sb-total');
    if (sbTotal) sbTotal.textContent = `${total} 任务 · ${running} 运行`;

    // 高度 = 段头(n * 24px) + 任务卡(n * ROW_H)
    let rows = 0;
    for (const s of segs) { rows += 1 + s.jobs.length; } // 1 行段头
    const H_PER_HEAD = 24;
    state.listSpacerEl.style.height = (segs.length * H_PER_HEAD + total * ROW_H) + 'px';
    state._segs = segs;
    if (state.currentJobId) {
      // 计算 current 在 spacer 中的 y 位置,需要时滚动居中
      let y = 0;
      for (const s of segs) {
        const i = s.jobs.findIndex(j => j.id === state.currentJobId);
        if (i >= 0) { y += H_PER_HEAD + i * ROW_H; break; }
        y += H_PER_HEAD + s.jobs.length * ROW_H;
      }
      const targetTop = y - state.listScrollEl.clientHeight / 2 + ROW_H / 2;
      const cur = state.listScrollEl.scrollTop;
      if (Math.abs(cur - targetTop) > ROW_H) state.listScrollEl.scrollTop = Math.max(0, targetTop);
    }
    renderVp();
  }

  /** 虚拟滚动:段头 + 任务卡,所有元素按 (segmentId,idx) 复用。 */
  function renderVp() {
    const segs = state._segs || [];
    const H_PER_HEAD = 24;
    const vp = state.listVpEl;
    const scrollTop = state.listScrollEl.scrollTop, vpH = state.listScrollEl.clientHeight;
    const overscan = 6;
    // 把 (segIdx, itemIdx, kind) 线性化以便算 y 区间
    const items = []; // {kind:'h'|'j', seg, job?, y}
    let y = 0;
    for (let s = 0; s < segs.length; s++) {
      items.push({ kind: 'h', label: segs[s].kind, y }); y += H_PER_HEAD;
      for (let i = 0; i < segs[s].jobs.length; i++) {
        items.push({ kind: 'j', seg: s, idx: i, job: segs[s].jobs[i], y }); y += ROW_H;
      }
    }
    const total = items.length;
    const start = Math.max(0, Math.floor((scrollTop - overscan * ROW_H) / ROW_H));
    const end = Math.min(total, Math.ceil((scrollTop + vpH + overscan * ROW_H) / ROW_H));
    const keep = new Set();
    for (let i = start; i < end; i++) {
      const it = items[i];
      const id = it.kind === 'h' ? `sb-h-${it.label}-${i}` : `sb-j-${it.job.id}`;
      keep.add(id);
      let el = vp.querySelector('#' + CSS.escape(id));
      if (!el) {
        el = it.kind === 'h' ? makeHead(it.label, i) : makeItem(it.job);
        vp.appendChild(el);
      }
      if (it.kind === 'h') el.style.transform = `translateY(${it.y}px)`;
      else {
        el.style.transform = `translateY(${it.y}px)`;
        updateItem(el, it.job);
      }
    }
    Array.from(vp.children).forEach(c => { if (!keep.has(c.id)) c.remove(); });
    if (total === 0) {
      if (!vp.querySelector('.sb-empty')) {
        const e = document.createElement('div');
        e.className = 'sb-empty';
        e.innerHTML = '<div class="sb-empty-mark">◐</div><div>暂无任务</div><div class="sb-empty-hint">点 + 新建开始识别</div>';
        vp.appendChild(e);
      }
    } else { const e = vp.querySelector('.sb-empty'); if (e) e.remove(); }
  }

  function makeHead(label, idx) {
    const el = document.createElement('div');
    el.className = 'sb-head-row';
    el.id = 'sb-h-' + label + '-' + idx;
    el.innerHTML = `<span class="sb-head-label"></span><span class="sb-head-rule"></span>`;
    el.querySelector('.sb-head-label').textContent = label;
    return el;
  }

  function makeItem(j) {
    const el = document.createElement('div');
    el.id = 'sb-j-' + j.id; el.className = 'sb-item';
    el.setAttribute('role', 'option');
    el.tabIndex = -1;
    const id = j.id;
    el.addEventListener('click', () => preview.open(id));
    el.addEventListener('contextmenu', (e) => {
      if (modalKit.anyOpen()) return;
      showJobCtxMenu(e, id);
    });
    el.innerHTML = `
      <div class="sb-thumb"></div>
      <div class="sb-info">
        <div class="sb-name"></div>
        <div class="sb-meta">
          <span class="sb-dot"></span>
          <span class="sb-status"></span><span class="sb-sep">·</span>
          <span class="sb-faces"></span>
        </div>
      </div>`;
    el._jobId = id;
    return el;
  }

  function itemSig(j) {
    const st = j.stats || {};
    return [
      j.id, j.status, j.display_name || '',
      j.face_count || 0,
      j.id === state.currentJobId ? 1 : 0,
      state.pinIds.has(j.id) ? 1 : 0,
      state.search,
    ].join('|');
  }

  function updateItem(el, j) {
    const sig = itemSig(j);
    if (el._sig === sig) return;
    el._sig = sig;
    el.classList.toggle('active', j.id === state.currentJobId);
    const st = j.stats || {};
    const fc = j.face_count || 0;
    const rawKey = j.cover_key || j.original_key;
    const thumbKey = rawKey && /\.(png|jpe?g|webp|gif|bmp)(\?|$)/i.test(rawKey) ? rawKey : null;
    const thumbSrc = thumbKey ? utils.mediaUrl(thumbKey) : null;
    const thumb = el.querySelector('.sb-thumb');
    if (thumb) {
      let inner;
      if (j.status === 'running' || j.status === 'queued') inner = '<div class="sb-thumb-spin">⏳</div>';
      else if (thumbSrc) inner = `<img data-src="${utils.escapeHtml(thumbSrc)}" alt="">`;
      else inner = '<div class="sb-thumb-empty">·</div>';
      if (thumb._inner !== inner) {
        thumb._inner = inner; thumb.innerHTML = inner;
        const img = thumb.querySelector('img[data-src]');
        if (img) {
          img.addEventListener('load', () => img.classList.add('is-loaded'), { once: true });
          img.addEventListener('error', () => img.classList.add('is-loaded'), { once: true });
          state.faceCardObserver.observe(img);
        }
      }
    }
    const nameEl = el.querySelector('.sb-name');
    if (nameEl) {
      const html = utils.highlight(j.display_name || j.id, state.search);
      if (nameEl._html !== html) { nameEl._html = html; nameEl.innerHTML = html; nameEl.title = j.display_name || j.id; }
    }
    const statusEl = el.querySelector('.sb-status');
    if (statusEl) statusEl.textContent = statusText(j.status);
    const facesEl = el.querySelector('.sb-faces');
    if (facesEl) facesEl.textContent = `${fc} 脸`;
    const dot = el.querySelector('.sb-dot');
    if (dot) dot.className = 'sb-dot ' + (j.status || '');
  }

  function statusText(s) {
    if (s === 'running') return '运行';
    if (s === 'queued') return '排队';
    if (s === 'done') return '完成';
    if (s === 'error') return '错误';
    if (s === 'cancelled') return '已停';
    return s || '—';
  }

  function showJobCtxMenu(e, jobId) {
    const job = (state.jobs || []).find(x => x.id === jobId);
    if (!job) return;
    const items = [
      { label: '打开任务', hint: 'Enter', onPick: () => preview.open(jobId) },
      { label: state.pinIds.has(jobId) ? '取消置顶' : '置顶任务', hint: state.pinIds.has(jobId) ? '📍' : '📌', onPick: () => togglePin(jobId) },
      { label: '复制任务 ID', hint: '⌗', onPick: () => {
          navigator.clipboard.writeText(job.id).then(
            () => toast.success('已复制任务 ID'),
            () => toast.error('复制失败'),
          );
        } },
    ];
    if (job.status === 'running' || job.status === 'queued') {
      items.push({ label: '取消任务', hint: '⏹', danger: true, onPick: () => cancelJobById(jobId) });
    }
    const canRetry = !!(job.original_input || (job.kind === 'image' && job.original_key));
    if (canRetry) items.push({ label: '重跑任务', hint: '↻', onPick: () => retryJobById(jobId) });
    items.push({ label: '删除任务', hint: '⌫', danger: true, onPick: () => deleteJobById(jobId) });
    ctxMenu.show(e, items);
  }

  function togglePin(id) {
    if (!id) return;
    if (state.pinIds.has(id)) { state.pinIds.delete(id); toast.info('已取消置顶'); }
    else { state.pinIds.add(id); toast.success('已置顶'); }
    try {
      localStorage.setItem(prefStore.KEYS.pinIds, JSON.stringify(Array.from(state.pinIds)));
    } catch {}
    utils.$$('.sb-item', state.listVpEl).forEach(el => { el._sig = null; });
    render();
  }

  function nextItem(dir) {
    const list = filtered(); if (!list.length) return;
    let i = list.findIndex(j => j.id === state.currentJobId);
    if (i < 0) i = 0; else i = Math.max(0, Math.min(list.length - 1, i + dir));
    preview.open(list[i].id);
  }

  function gotoOffset(offset) {
    const list = filtered(); if (!list.length) return;
    let i;
    if (offset === 0) i = 0;
    else if (offset < 0) i = list.length - 1;
    else i = Math.min(list.length - 1, Math.max(0, offset));
    preview.open(list[i].id);
  }

  return { init, setJobs, upsertJob, render, setActive: () => {}, nextItem, gotoOffset, filtered, togglePin, statusText };
})();

/** 详情预览:单视频 + 单图渲染 + 标注叠加 + 进度 + 内联一行统计 + 人脸卡片。 */
const preview = (() => {
  let lastFrameIdx = -1, _lastAnnoFrameIdx = -1;
  let _detDims = {}; // jobId -> {w,h}

  function showEmpty() { _faceGridSig = null; utils.$('#pv-empty').classList.remove('hidden'); utils.$('#pv-detail').classList.add('hidden'); }
  function showDetail() { utils.$('#pv-empty').classList.add('hidden'); utils.$('#pv-detail').classList.remove('hidden'); }
  const showHint = t => { const h = utils.$('#pv-stage-hint'); h.textContent = t; h.classList.remove('hidden'); };
  const hideHint = () => utils.$('#pv-stage-hint').classList.add('hidden');

  function resetMedia() {
    const img = utils.$('#pv-img'); if (img) { img.removeAttribute('src'); img.classList.add('hidden'); img.classList.remove('is-loaded'); }
    const vid = utils.$('#pv-vid'); if (vid) { vid.removeAttribute('src'); vid.classList.add('hidden'); }
    clearOverlay();
    const banner = utils.$('#pv-error-banner'); if (banner) {
      banner.classList.add('hidden');
      const msgEl = banner.querySelector('.pv-error-msg');
      const actEl = banner.querySelector('.pv-error-act');
      if (msgEl) msgEl.textContent = '';
      if (actEl) actEl.innerHTML = '';
    }
  }

  async function open(jobId) {
    state.currentJobId = jobId; lastFrameIdx = -1; _lastAnnoFrameIdx = -1;
    showDetail(); showHint('加载中…'); resetMedia(); sse.detach();
    try {
      const job = await api.getJob(jobId);
      state.currentJob = job; render(job);
      if (job.status === 'running' || job.status === 'queued') sse.attach(jobId);
    } catch (e) { toast.error('加载任务失败: ' + e.message); }
  }

  function close() {
    state.currentJobId = null; state.currentJob = null;
    sse.detach(); showEmpty(); resetMedia();
  }

  function render(job) {
    // 切换到另一个 job 时旧的簇指纹不再适用,强制下一次重建网格。
    if (state.currentJobId && job && job.id && state.currentJobId !== job.id) _faceGridSig = null;
    utils.$('#pv-title').textContent = job.display_name || job.id;
    const idEl = utils.$('#pv-id');
    if (idEl) idEl.textContent = '#' + (job.id || '').slice(0, 8);
    const st = job.stats || {};
    const totalFrames = st.total_frames || (job.kind === 'image' ? job.frame_count : st.frames_processed) || st.frames_processed || 0;
    utils.$('#pv-faces').textContent = `${job.face_count || 0} 张脸`;
    utils.$('#pv-time').textContent = utils.fmtAbsTime(job.created_ms);
    const status = job.status || 'queued';
    const statusEl = utils.$('#pv-status'); statusEl.textContent = sidebar.statusText(status); statusEl.className = 'pv-status ' + status;
    utils.$('#pv-dot').className = 'pv-dot ' + status;

    // GPU 行(取 config 里的 embedder device)
    const gpuEl = utils.$('#pv-gpu');
    if (gpuEl && state.config && state.config.embedder_device) gpuEl.textContent = state.config.embedder_device;

    // 错误 banner
    const banner = utils.$('#pv-error-banner');
    if (status === 'error' || status === 'cancelled') {
      const label = status === 'cancelled' ? '任务已取消' : '识别失败';
      const msg = job.error ? `${label} · ${job.error}` : label;
      if (banner) {
        banner.classList.remove('hidden');
        banner.querySelector('.pv-error-msg').textContent = msg;
        const actEl = banner.querySelector('.pv-error-act');
        actEl.innerHTML = '';
        if (job.error) {
          const copy = document.createElement('button');
          copy.className = 'pv-btn ghost';
          copy.textContent = '复制错误';
          copy.addEventListener('click', () => {
            navigator.clipboard.writeText(`[${job.display_name || job.id}] ${job.error}`).then(() => toast.success('已复制'), () => toast.error('复制失败'));
          });
          actEl.appendChild(copy);
        }
        const canRetry = job.original_input || (job.kind === 'image' && job.original_key);
        if (canRetry) {
          const retry = document.createElement('button');
          retry.className = 'pv-btn primary';
          retry.textContent = '↻ 重跑';
          retry.addEventListener('click', retryCurrent);
          actEl.appendChild(retry);
        }
      }
    } else if (banner) banner.classList.add('hidden');

    if (job.kind === 'image') renderImage(job);
    else if (job.kind === 'video') renderVideo(job);
    else renderStream(job);
    renderProgress(job);
    renderFaceGrid(job);
  }

  /** FPS 估算:对当前显示的实时帧率做 EMA。终态用 elapsed_ms 算平均 fps。 */
  function updatePerfMetrics(job) {
    const st = job.stats || {};
    const now = Date.now();
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
    } else {
      state._fpsLastAt = 0; state._fpsSamples.length = 0;
    }
  }

  function renderImage(job) {
    const img = utils.$('#pv-img'); if (!img) return;
    const vid = utils.$('#pv-vid'); if (vid) vid.classList.add('hidden');
    img.classList.remove('hidden');
    const frames = job.frames || [];
    const first = frames.find(f => f.original_key) || frames[0];
    const src = (first && first.original_key) || job.original_key;
    if (src) {
      img.onload = () => {
        hideHint();
        img.classList.add('is-loaded');
        if (first && first.annotated_key) drawOverlay(job, first, img);
      };
      img.onerror = () => { showHint('原图加载失败'); clearOverlay(); };
      img.src = utils.mediaUrl(src);
    } else { img.classList.add('hidden'); showHint('无原图'); clearOverlay(); }
  }

  function renderVideo(job) {
    const img = utils.$('#pv-img'); if (img) img.classList.add('hidden');
    const vid = utils.$('#pv-vid'); if (!vid) return;
    vid.classList.remove('hidden');
    ensureDetDims(job);
    if (!job.original_key) { showHint('无视频源'); return; }
    const newSrc = utils.mediaUrl(job.original_key);
    if (!vid.src || !vid.src.includes(encodeURIComponent(job.original_key))) vid.src = newSrc;
    const first = (job.frames || []).find(f => f.annotated_key);
    vid.poster = first ? utils.mediaUrl(first.annotated_key) : '';
    vid.onloadedmetadata = () => { hideHint(); syncVideoOverlay(job, vid); };
    vid.ontimeupdate = () => syncVideoOverlay(job, vid);
    vid.onseeked = () => syncVideoOverlay(job, vid);
    syncVideoOverlay(job, vid);
  }

  function ensureDetDims(job) {
    if (_detDims[job.id]) return;
    const first = (job.frames || []).find(f => f.annotated_key);
    if (!first) return;
    const probe = new Image();
    probe.onload = () => {
      _detDims[job.id] = { w: probe.naturalWidth, h: probe.naturalHeight };
      if (state.currentJob && state.currentJob.id === job.id) {
        const cur = (job.frames || []).find(f => f.index === _lastAnnoFrameIdx);
        const vid = utils.$('#pv-vid');
        if (cur && vid && !vid.classList.contains('hidden')) drawVideoOverlay(job, cur, vid);
      }
    };
    probe.src = utils.mediaUrl(first.annotated_key);
  }

  function syncVideoOverlay(job, vid) {
    const frames = job.frames || []; if (!frames.length) return;
    const cur = vid.currentTime * 1000;
    let best = null;
    for (const f of frames) {
      if (f.annotated_key && f.timestamp_ms <= cur + 80) best = f;
      else if (f.timestamp_ms > cur + 80) break;
    }
    if (best && cur - best.timestamp_ms <= 400) {
      if (best.index !== _lastAnnoFrameIdx) _lastAnnoFrameIdx = best.index;
      drawVideoOverlay(job, best, vid);
    } else if ((!best || cur - best.timestamp_ms > 400) && _lastAnnoFrameIdx !== -1) {
      _lastAnnoFrameIdx = -1;
      clearOverlay();
    }
  }

  function renderStream(job) {
    const img = utils.$('#pv-img');
    const vid = utils.$('#pv-vid'); if (vid) vid.classList.add('hidden');
    if (!img) return;
    img.classList.remove('hidden');
    const frames = job.frames || [];
    const last = [...frames].reverse().find(f => f.original_key) || frames[frames.length - 1];
    if (!last) {
      if (job.original_key) {
        img.onload = () => hideHint();
        img.onerror = () => showHint('原图加载失败');
        img.src = utils.mediaUrl(job.original_key);
      } else { img.classList.add('hidden'); showHint('等待原始帧…'); clearOverlay(); }
      return;
    }
    if (last.index === lastFrameIdx) return;
    lastFrameIdx = last.index;
    img.onload = () => {
      hideHint();
      img.classList.add('is-loaded');
      drawOverlay(job, last, img);
    };
    img.onerror = () => { showHint('原图加载失败'); };
    img.classList.remove('is-loaded');
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
    c.style.width = mediaRect.width + 'px'; c.style.height = mediaRect.height + 'px';
    c.width = mediaRect.width; c.height = mediaRect.height;
    drawBoxes(c, frame, mediaEl.naturalWidth || mediaEl.videoWidth, mediaEl.naturalHeight || mediaEl.videoHeight);
  }

  function drawVideoOverlay(job, frame, vid) {
    if (!state.annoVisible) return clearOverlay();
    const c = utils.$('#pv-overlay'), stage = utils.$('#pv-stage');
    if (!c || !stage) return;
    const rect = stage.getBoundingClientRect(), m = vid.getBoundingClientRect();
    c.style.left = (m.left - rect.left) + 'px';
    c.style.top = (m.top - rect.top) + 'px';
    c.style.width = m.width + 'px'; c.style.height = m.height + 'px';
    c.width = m.width; c.height = m.height;
    const dims = _detDims[job.id];
    const natW = dims ? dims.w : (frame.faces[0] ? frame.faces[0].w * 2 : 480);
    const natH = dims ? dims.h : 270;
    drawBoxes(c, frame, natW, natH);
  }

  function drawBoxes(canvas, frame, natW, natH) {
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
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
          if (f) drawVideoOverlay(job, f, vid);
        }
      }
    });
  });

  function renderProgress(job) {
    const st = job.stats || {};
    const processed = st.frames_processed || 0;
    const total = st.total_frames || (job.kind === 'image' ? job.frame_count : processed) || processed;
    const pct = total > 0 ? Math.min(100, (processed / total) * 100) : 0;
    utils.$('#pv-prog-fill').style.width = pct + '%';
    utils.$('#pv-prog-text').textContent = `${processed} / ${total}`;
    utils.$('#pv-prog-pct').textContent = pct.toFixed(0) + '%';
  }

  /**
   * 人脸网格上次渲染的簇指纹(见 renderFaceGrid)。job 切换时置 null。
   */
  let _faceGridSig = null;

  /**
   * 人脸网格:按 timestamp 把 < 2s 的检出聚成"同一人",每簇取中位帧作代表。
   *
   * 性能:这是 SSE `frame` 事件里最贵的一段 —— 每次都要遍历全部 frame×face,
   * 重建全部 DOM 卡片,并给每张 <img> 重新注册 IntersectionObserver。
   * 长视频下 frames 会累积到几千,一帧事件就触发一次全量重建。
   *
   * 优化:帧事件只会在末尾**追加**数据,已渲染的簇几乎不会改变。所以先算出
   * "代表帧 key"这一串指纹,和上次渲染的一致就直接返回 —— 绝大多数事件
   * 走这条 cheap path,不动 DOM。指纹真变了才重建。
   */
  function renderFaceGrid(job) {
    const grid = utils.$('#pv-faces-grid'); if (!grid) return;
    const faces = [];
    for (const f of job.frames || []) {
      for (const face of (f.faces || [])) {
        faces.push({ ts: f.timestamp_ms, key: face.key, score: face.score, frame: f, x: face.x, y: face.y, w: face.w, h: face.h });
      }
    }
    if (!faces.length) {
      grid.innerHTML = '<div class="empty">尚未出现人脸</div>';
      _faceGridSig = null;
      return;
    }
    // 简版 cluster:按 ts 增量 < 2s 视为同一人
    const clusters = [];
    let cur = [faces[0]];
    for (let i = 1; i < faces.length; i++) {
      const dt = faces[i].ts - faces[i - 1].ts;
      if (dt >= 0 && dt < 2000) cur.push(faces[i]);
      else { clusters.push(cur); cur = [faces[i]]; }
    }
    clusters.push(cur);

    // 指纹 = 每簇代表帧的 key + 簇大小。追加新帧时,只有最后一簇会变,
    // 前面所有簇的代表帧保持不变 —— 但簇大小变了(xN badge),所以也要计入。
    const sig = clusters.map(cl => {
      const rep = cl[Math.floor(cl.length / 2)];
      return rep.key + '#' + cl.length;
    }).join('|');
    if (sig === _faceGridSig) return;   // 无变化,跳过 DOM 重建
    _faceGridSig = sig;

    const frag = document.createDocumentFragment();
    for (const cl of clusters) {
      const rep = cl[Math.floor(cl.length / 2)];
      const card = document.createElement('div');
      card.className = 'face-card';
      card.innerHTML = `
        <div class="fc-img-wrap">
          <div class="fc-skel"></div>
          <img loading="lazy" decoding="async" data-src="${utils.escapeHtml(utils.mediaUrl(rep.key))}" alt="人脸">
          ${rep.frame ? `<div class="fc-idx">#${rep.frame.index}</div>` : ''}
          <div class="fc-time">⏱ ${utils.fmtTime(rep.ts)}</div>
          ${cl.length > 1 ? `<div class="fc-badge">×${cl.length}</div>` : ''}
        </div>
        <div class="fc-score">score <span>${rep.score.toFixed(2)}</span></div>`;
      const img = card.querySelector('img');
      if (img) img.addEventListener('load', () => {
        img.classList.add('is-loaded');
        const skel = card.querySelector('.fc-skel');
        if (skel) skel.classList.add('is-done');
      });
      img.addEventListener('error', () => {
        img.classList.add('is-loaded');
        const skel = card.querySelector('.fc-skel');
        if (skel) skel.classList.add('is-done');
      });
      card.addEventListener('click', () => seekToFrame(rep.frame));
      frag.appendChild(card);
      // 直接 src + loading=lazy:浏览器原生按视口加载。
      // 原 IntersectionObserver 以 #sb-scroll 为 root,face card 在主区根本不会 intersect,
      // 导致 src 永远是空,只有灰色骨架。
      img.src = utils.mediaUrl(rep.key);
    }
    grid.replaceChildren(frag);
  }

  function seekToFrame(frame) {
    if (!frame) return;
    const job = state.currentJob; if (!job) return;
    if (job.kind === 'video') {
      const v = utils.$('#pv-vid'); if (v) { v.currentTime = frame.timestamp_ms / 1000; v.play().catch(() => {}); }
    }
  }

  function toggleAnno() {
    state.annoVisible = !state.annoVisible;
    prefStore.write(prefStore.KEYS.annoVisible, state.annoVisible ? '1' : '0');
    if (state.currentJob) {
      if (state.annoVisible) render(state.currentJob);
      else clearOverlay();
    }
  }

  return { open, close, render, showEmpty, showDetail, toggleAnno, updatePerfMetrics };
})();

/** SSE:服务端主动推 frame / heartbeat / done 事件。 */
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
      if (msg.type === 'frame') {
        if (state.currentJob && msg.frame) {
          if (!Array.isArray(state.currentJob.frames)) state.currentJob.frames = [];
          const idx = msg.frame.index;
          const dup = state.currentJob.frames.some(f => f && f.index === idx);
          if (!dup) state.currentJob.frames.push(msg.frame);
          if (msg.inline) state.currentInline = msg.inline;
          preview.render(state.currentJob);
        }
        scheduleRefresh();
      }
      else if (msg.type === 'heartbeat') {
        preview.updatePerfMetrics(state.currentJob || {});
        scheduleRefresh();
      }
      else if (msg.type === 'done' || msg.type === 'cancelled' || msg.type === 'error') {
        detach();
        apiCache.bust('listJobs');
        scheduleRefresh();
        if (api && api.listJobs && sidebar && sidebar.setJobs) {
          api.listJobs().then(jobs => { try { sidebar.setJobs(jobs); } catch {} }).catch(() => {});
        }
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

/** 上传:统一入口(图片 / 视频 / 流)— 走 uploadQueue。 */
const upload = (() => {
  function init() {
    const on = (sel, fn) => { const el = utils.$(sel); if (el) el.addEventListener('click', fn); };
    on('#tb-home', () => preview.close());
    on('#pv-empty-new', () => openModal());
    on('#pv-kebab', (e) => {
      if (!state.currentJob) return;
      const job = state.currentJob;
      const items = [
        { label: state.annoVisible ? '隐藏标注' : '显示标注', hint: 'A', onPick: () => preview.toggleAnno() },
        { label: '复制任务 ID', hint: '⌗', onPick: () => {
            navigator.clipboard.writeText(job.id).then(() => toast.success('已复制'), () => toast.error('复制失败'));
          } },
      ];
      if (job.status === 'running' || job.status === 'queued') items.push({ label: '取消任务', hint: '⏹', danger: true, onPick: () => cancelCurrent() });
      const canRetry = !!(job.original_input || (job.kind === 'image' && job.original_key));
      if (canRetry) items.push({ label: '重跑任务', hint: '↻', onPick: () => retryCurrent() });
      items.push({ label: '导出 JSON', hint: 'JSON', onPick: () => exportJSON() });
      items.push({ label: '删除任务', hint: '⌫', danger: true, onPick: () => deleteCurrent() });
      ctxMenu.show(e, items, e.currentTarget);
    });
    // 单一 dropzone:接受所有(图片 + 视频),按 MIME 分发
    setupDropzone('#dz-unified', '#file-unified', files => {
      const imgs = [], vids = [];
      for (const f of files) {
        if (f.type && f.type.startsWith('image/')) imgs.push(f);
        else if (f.type && f.type.startsWith('video/')) vids.push(f);
        else if (/\.(jpe?g|png|webp|bmp|gif)$/i.test(f.name || '')) imgs.push(f);
        else if (/\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(f.name || '')) vids.push(f);
      }
      if (imgs.length) uploadQueue.enqueue(imgs, 'image');
      if (vids.length) uploadQueue.enqueue([vids[0]], 'video'); // 视频一次一个
      if (!imgs.length && !vids.length) toast.warn('未识别到可处理文件');
      closeAllModals();
    });
    // URL 输入行:支持 http/https/m3u8 → 视频流 / rtsp → 流任务
    const urlGo = utils.$('#unified-url');
    if (urlGo) urlGo.addEventListener('keydown', e => {
      if (e.key === 'Enter') {
        e.preventDefault();
        const url = urlGo.value.trim(); if (!url) return;
        if (/^rtsp:\/\//i.test(url)) submitStream(url);
        else submitVideoUrl(url);
        closeAllModals();
      }
    });
    setupGlobalDrop();
    setupGlobalPaste();
  }

  function openModal() { utils.$('#modal-new').classList.remove('hidden'); setTimeout(() => utils.$('#unified-url')?.focus(), 50); }
  function closeAllModals() { modalKit.closeAll(); }

  function submitVideoUrl(url) {
    toast.info(`正在拉取 ${url.slice(0, 60)}…`);
    api.importVideoUrl(url).then(data => {
      if (data.error) return toast.error('导入失败: ' + utils.explainError(data, '服务端错误'));
      apiCache.bustAll();
      toast.success('已拉取,开始识别 #' + (data.job_id || '').slice(0, 8));
      api.getJob(data.job_id).then(job => { sidebar.upsertJob(job); preview.open(job.id); });
    }).catch(e => toast.error('导入失败: ' + e.message));
  }

  function submitStream(url) {
    api.postStream(url).then(data => {
      if (data.error) return toast.error('启动失败: ' + data.error);
      apiCache.bustAll();
      toast.success('已启动流 #' + (data.job_id || '').slice(0, 8));
      api.getJob(data.job_id).then(job => { sidebar.upsertJob(job); preview.open(job.id); });
    }).catch(e => toast.error('启动失败: ' + e.message));
  }

  function setupDropzone(zoneSel, inputSel, handler) {
    const zone = utils.$(zoneSel), input = utils.$(inputSel); if (!zone || !input) return;
    zone.addEventListener('click', e => { if (!e.target.closest('input')) input.click(); });
    input.addEventListener('change', () => { if (input.files.length) handler(Array.from(input.files)); input.value = ''; });
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', e => { if (!zone.contains(e.relatedTarget)) zone.classList.remove('dragover'); });
    zone.addEventListener('drop', async e => {
      e.preventDefault(); e.stopPropagation(); zone.classList.remove('dragover');
      if (!e.dataTransfer) return;
      const collected = await dropzonePreview.collectFromDataTransfer(e.dataTransfer);
      if (!collected.length) return toast.warn('未识别到文件,请重试');
      handler(collected.map(c => c.file));
    });
  }

  function setupGlobalDrop() {
    let dragDepth = 0;
    const showOverlay = () => utils.$('#dropzone-overlay')?.classList.remove('hidden');
    const hideOverlay = () => utils.$('#dropzone-overlay')?.classList.add('hidden');
    document.addEventListener('dragenter', e => {
      if (!e.dataTransfer || Array.from(e.dataTransfer.types || []).indexOf('Files') < 0) return;
      e.preventDefault(); dragDepth++;
      if (!utils.$('#modal-new').classList.contains('hidden')) return;
      showOverlay();
    });
    document.addEventListener('dragover', e => { if (e.dataTransfer) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; } });
    document.addEventListener('dragleave', () => { dragDepth = Math.max(0, dragDepth - 1); if (dragDepth === 0) hideOverlay(); });
    document.addEventListener('drop', async e => {
      e.preventDefault(); dragDepth = 0; hideOverlay();
      if (!e.dataTransfer) return;
      if (!utils.$('#modal-new').classList.contains('hidden')) return;
      const collected = await dropzonePreview.collectFromDataTransfer(e.dataTransfer);
      const imgs = collected.filter(c => c.file.type?.startsWith('image/') || /\.(jpe?g|png|webp|bmp|gif)$/i.test(c.file.name));
      const vids = collected.filter(c => c.file.type?.startsWith('video/') || /\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(c.file.name));
      if (imgs.length) uploadQueue.enqueue(imgs.map(c => c.file), 'image');
      if (vids.length) uploadQueue.enqueue([vids[0].file], 'video');
      if (!imgs.length && !vids.length) toast.warn('目录中没有可识别的图片/视频');
    });
  }

  function setupGlobalPaste() {
    document.addEventListener('paste', async e => {
      if (e.target && e.target.matches && e.target.matches('input, textarea, [contenteditable]')) return;
      const cd = e.clipboardData; if (!cd) return;
      for (const it of cd.items || []) {
        if (it.kind === 'file' && it.type && it.type.startsWith('image/')) {
          const f = it.getAsFile(); if (f) { e.preventDefault(); uploadQueue.enqueue([f], 'image'); return; }
        }
      }
      const text = (cd.getData && cd.getData('text')) || '';
      const url = text.trim();
      if (/^https?:\/\/\S+$/i.test(url) || /^rtsp:\/\/\S+$/i.test(url)) {
        if (!await confirmModal.open('粘贴 URL', `创建 ${/^rtsp/.test(url) ? '直播流' : '视频'} 任务?\n${url.slice(0, 100)}${url.length > 100 ? '…' : ''}`, { okText: '创建' })) return;
        if (/^rtsp/.test(url)) submitStream(url); else submitVideoUrl(url);
      }
    });
  }

  function loadConfig() {
    return api.getConfig().then(cfg => {
      state.config = cfg || {};
      const dev = state.config.embedder_device || '';
      const el = utils.$('#tb-runtime');
      if (el) {
        if (!dev) { el.hidden = true; el.textContent = ''; el.removeAttribute('data-state'); }
        else {
          el.hidden = false;
          el.textContent = dev;
          let s = 'cpu';
          if (dev.includes('cuda') || dev.includes('tensorrt')) s = 'gpu';
          else if (dev === 'coreml' || dev === 'directml') s = 'accel';
          el.setAttribute('data-state', s);
        }
      }
    }).catch(() => {});
  }

  return { init, openModal, loadConfig, submitVideoUrl, submitStream };
})();

/** 帮助 modal:`?` 键 / 状态栏按钮 / Esc 关闭。 */
const helpModal = (() => {
  function isOpen() {
    const m = utils.$('#modal-help');
    return !!(m && !m.classList.contains('hidden'));
  }
  function init() {
    const btn = utils.$('#sb-help');
    if (btn) btn.addEventListener('click', toggle);
  }
  function open() {
    const m = utils.$('#modal-help'); if (!m) return;
    m.classList.remove('hidden');
  }
  function close() {
    const m = utils.$('#modal-help'); if (!m) return;
    m.classList.add('hidden');
  }
  function toggle() {
    if (isOpen()) close(); else open();
  }
  return { init, open, close, toggle, isOpen };
})();

/** 算法选择器:新建任务 modal 角落的 chip 行,从 /api/config 取可用列表,
 *  选择持久化到 localStorage,供 uploadQueue 提交时读取。 */
const algoPicker = (() => {
  const KEY = 'rsface.algo';
  let current = '';
  let available = [];
  const ALGO_LABEL = {
    haar: 'Haar',
    cnn: 'CNN',
    luminance: 'Luminance',
    skin: 'Skin',
    lbp: 'LBP',
    hog: 'HOG',
  };

  function read() {
    try { return localStorage.getItem(KEY) || ''; } catch { return ''; }
  }
  function write(v) {
    try { if (v) localStorage.setItem(KEY, v); else localStorage.removeItem(KEY); } catch {}
  }
  function refreshLabel() {
    const cur = utils.$('#algo-current');
    if (cur) cur.textContent = current ? '· ' + (ALGO_LABEL[current] || current) : '';
  }
  function render() {
    const panel = utils.$('#algo-panel');
    if (!panel) return;
    const items = [{ id: '', name: '默认' }].concat(
      available.map(a => ({ id: a, name: ALGO_LABEL[a] || a }))
    );
    panel.innerHTML = items.map(it =>
      `<button type="button" class="algo-chip${it.id === current ? ' is-on' : ''}" data-algo="${utils.escapeHtml(it.id)}">${utils.escapeHtml(it.name)}</button>`
    ).join('');
    panel.querySelectorAll('.algo-chip').forEach(b => {
      b.addEventListener('click', () => select(b.dataset.algo));
    });
    refreshLabel();
  }
  function setAvailable(list) {
    available = Array.isArray(list) ? list.slice() : [];
    // 若上次选择的算法已下线,清空
    if (current && !available.includes(current)) { current = ''; write(''); }
    render();
  }
  function select(algo) {
    current = algo || '';
    write(current);
    render();
    const btn = utils.$('#algo-link');
    if (btn) btn.setAttribute('aria-expanded', isOpen() ? 'true' : 'false');
  }
  function isOpen() {
    const p = utils.$('#algo-panel');
    return !!(p && !p.hidden);
  }
  function open() {
    const p = utils.$('#algo-panel'); if (!p) return;
    p.hidden = false;
    const btn = utils.$('#algo-link'); if (btn) btn.setAttribute('aria-expanded', 'true');
  }
  function close() {
    const p = utils.$('#algo-panel'); if (!p) return;
    p.hidden = true;
    const btn = utils.$('#algo-link'); if (btn) btn.setAttribute('aria-expanded', 'false');
  }
  function toggle() { if (isOpen()) close(); else open(); }
  function init() {
    current = read();
    const btn = utils.$('#algo-link');
    if (btn) btn.addEventListener('click', e => { e.preventDefault(); toggle(); });
    refreshLabel();
    // 新建任务 modal 关闭时,自动收起面板,避免脏状态
    const modalNew = utils.$('#modal-new');
    if (modalNew) {
      new MutationObserver(() => {
        if (modalNew.classList.contains('hidden')) close();
      }).observe(modalNew, { attributes: true, attributeFilter: ['class'] });
    }
    // 立即用兜底列表渲染(jsdom / fetch 失败时仍可见),再用 /api/config 升级
    setAvailable(['haar', 'cnn', 'luminance', 'skin', 'lbp', 'hog']);
    fetch('/api/config').then(r => r.ok ? r.json() : null).then(cfg => {
      if (cfg && Array.isArray(cfg.available_algos)) setAvailable(cfg.available_algos);
    }).catch(() => {});
  }
  function get() { return current || undefined; }
  // 暴露给 uploadQueue.js(不同脚本,无法直接闭包访问)
  window.__algoPicker = { get, select };
  return { init, get, select, open, close, toggle, isOpen };
})();

/** 全局快捷键 + 任务级操作(cancel / retry / 导出 JSON)。 */
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
    if ((e.ctrlKey || e.metaKey) && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault();
      const s = utils.$('#tb-search'); if (s) { s.focus(); s.select(); }
      return;
    }
    if (e.key === 'n' || e.key === 'N') { upload.openModal(); e.preventDefault(); return; }
    if (e.key === 'ArrowDown') { sidebar.nextItem(1); e.preventDefault(); return; }
    if (e.key === 'ArrowUp')   { sidebar.nextItem(-1); e.preventDefault(); return; }
    if (e.key === 'j' || e.key === 'J') { sidebar.nextItem(1); e.preventDefault(); return; }
    if (e.key === 'k' || e.key === 'K') { sidebar.nextItem(-1); e.preventDefault(); return; }
    if (e.key === 'Home') { sidebar.gotoOffset(0); e.preventDefault(); return; }
    if (e.key === 'End')  { sidebar.gotoOffset(-1); e.preventDefault(); return; }
    if (e.key === 'PageDown' || e.key === 'PageUp') {
      const rowH = state.itemHeight + state.itemGap;
      const step = Math.max(1, Math.floor((state.listScrollEl ? state.listScrollEl.clientHeight : 400) / rowH) - 1);
      const list = sidebar.filtered();
      const cur = list.findIndex(j => j.id === state.currentJobId);
      if (cur < 0) sidebar.gotoOffset(0);
      else if (e.key === 'PageDown') sidebar.gotoOffset(cur + step);
      else sidebar.gotoOffset(cur - step);
      e.preventDefault(); return;
    }
    if (e.key === 'a' || e.key === 'A') { preview.toggleAnno(); e.preventDefault(); return; }
    if (e.key === '?' || (e.key === '/' && e.shiftKey)) { helpModal.toggle(); e.preventDefault(); return; }
    if ((e.key === 'Delete' || e.key === 'Backspace') && state.currentJobId) { deleteCurrent(); e.preventDefault(); return; }
  });
}

function cancelCurrent() {
  if (!state.currentJobId) return;
  api.cancelJob(state.currentJobId).then(() => toast.info('已发送停止信号')).catch(e => toast.error('取消失败: ' + e.message));
}

async function cancelJobById(id) {
  if (!id) return;
  try { await api.cancelJob(id); toast.info('已发送停止信号'); }
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
    state.pinIds.delete(id);
    try { localStorage.setItem(prefStore.KEYS.pinIds, JSON.stringify(Array.from(state.pinIds))); } catch {}
    if (state.currentJobId === id) preview.close();
    sidebar.render();
    toast.success('任务已删除');
  } catch (e) { toast.error('删除失败: ' + e.message); }
}

async function deleteJobById(id) {
  if (!id) return;
  const job = (state.jobs || []).find(x => x.id === id);
  const name = job ? (job.display_name || id) : id;
  if (!await confirmModal.open('删除任务', `确认删除 "${name}"?此操作不可恢复。`)) return;
  try {
    await api.deleteJob(id);
    state.deletedIds.add(id);
    state.pinIds.delete(id);
    try { localStorage.setItem(prefStore.KEYS.pinIds, JSON.stringify(Array.from(state.pinIds))); } catch {}
    if (state.currentJobId === id) preview.close();
    sidebar.render();
    toast.success('任务已删除');
  } catch (e) { toast.error('删除失败: ' + e.message); }
}

async function retryCurrent() {
  if (!state.currentJobId) return;
  if (!await confirmModal.open('重跑任务', '将用原 URL/输入重新创建一个任务。继续?')) return;
  try {
    const r = await api.retryJob(state.currentJobId);
    toast.success('已重跑,新任务 #' + (r.job_id || '').slice(0, 8));
    if (r.job_id) {
      const job = await api.getJob(r.job_id);
      sidebar.upsertJob(job); preview.open(job.id);
    }
  } catch (e) { toast.error('重跑失败: ' + e.message); }
}

async function retryJobById(id) {
  if (!id) return;
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
  const blob = new Blob([JSON.stringify(summary, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = `job-${job.id}.json`;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 500);
  toast.success(`已下载 JSON (${summary.frame_count} 帧)`);
}

async function init() {
  theme.init(); modalKit.init();
  if (typeof uploadQueue !== 'undefined' && uploadQueue && typeof uploadQueue.init === 'function') uploadQueue.init();
  if (typeof dropzonePreview !== 'undefined' && dropzonePreview && typeof dropzonePreview.init === 'function') dropzonePreview.init();
  sidebar.init(); upload.init(); confirmModal.init();
  algoPicker.init(); helpModal.init(); initKeys();
  if (typeof visibilityCtl !== 'undefined' && visibilityCtl && typeof visibilityCtl.init === 'function') visibilityCtl.init();
  // 抽屉(窄屏)
  const drawerBtn = utils.$('#tb-drawer');
  const drawerBack = utils.$('#drawer-backdrop');
  if (drawerBtn && drawerBack) {
    const setDrawer = open => {
      utils.$('#sidebar').classList.toggle('drawer-open', open);
      drawerBack.classList.toggle('hidden', !open);
      drawerBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
    };
    drawerBtn.addEventListener('click', () => setDrawer(!utils.$('#sidebar').classList.contains('drawer-open')));
    drawerBack.addEventListener('click', () => setDrawer(false));
    window.addEventListener('resize', () => { if (window.innerWidth > 768) setDrawer(false); });
    drawerBtn.hidden = false;
  }
  try {
    const jobs = await api.listJobs();
    sidebar.setJobs(jobs);
  } catch (e) { toast.error('加载任务列表失败: ' + e.message); }
  upload.loadConfig().catch(() => {});
}

document.addEventListener('DOMContentLoaded', init);
window.__rsface = {
  state, api, sidebar, preview, sse, theme, toast,
  modalKit, confirmModal, ctxMenu,
  uploadQueue: (typeof uploadQueue !== 'undefined') ? uploadQueue : null,
  dropzonePreview: (typeof dropzonePreview !== 'undefined') ? dropzonePreview : null,
  visibilityCtl: (typeof visibilityCtl !== 'undefined') ? visibilityCtl : null,
};