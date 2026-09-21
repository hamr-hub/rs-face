/* rs-face / platform/web/recognize.js
 *
 * Multi-recogniser identity consensus panel.
 *
 * Loaded via index.html (defer, after compare.js).
 *
 * What it does:
 *  1. The topbar button (id=tb-recognize) toggles identity mode and the
 *     choice is persisted to localStorage.
 *  2. When on, the module calls
 *       POST /api/jobs/{id}/recognize?detector=haar
 *     for the currently-open image job.
 *  3. It renders a panel listing every detected face with its consensus
 *     identity (label, votes, confidence, agreeing recognisers) and the
 *     per-recogniser vote detail.
 *  4. Detected boxes with their consensus labels are drawn over the
 *     source image on a canvas.
 *
 * Zero new JS dependencies. DOM + canvas only.
 */
(() => {
  'use strict';

  const STORAGE_KEY = 'rsface.recognize.enabled';

  const RECOGNIZER_META = {
    lbph:       { label: 'LBPH',   color: '#5ec8ff' },
    eigenface:  { label: 'EIGEN',  color: '#b08bff' },
    fisherface: { label: 'FISHER', color: '#4fd6a0' },
  };

  const state = {
    lastJobId: null,
    pollTimer: null,
    /// 最近一次响应序列化后的指纹(face_count + 每个 face 的 (x,y,w,h,label,votes))。
    /// 同指纹就不重画 — 前端 600ms 轮询期间后端命中 per-job 缓存,
    /// 响应字节完全相同,避免每 tick 重绘 canvas + 重建 DOM(单图 ~50 节点)。
    lastResponseKey: '',
    /// 记录识别错误(404/500/no_gallery 等),避免每次轮询都打 toast。
    lastErrorKey: '',
    /// 终态(no_gallery / no_such_job)时停止轮询,避免无意义请求。
    pollingStopped: false,
  };

  function isEnabled() {
    try { return localStorage.getItem(STORAGE_KEY) === '1'; } catch { return false; }
  }
  function setEnabled(v) {
    try { localStorage.setItem(STORAGE_KEY, v ? '1' : '0'); } catch {}
  }

  function recognizerMeta(name) {
    return RECOGNIZER_META[name] || { label: String(name).toUpperCase(), color: '#d0d0d0' };
  }

  function injectStyles() {
    if (document.getElementById('rsfr-styles')) return;
    const css = `
      .rsfr-panel {
        margin: 12px 0;
        background: var(--bg-3, #181d27);
        border: 1px solid var(--border, rgba(255,255,255,0.08));
        border-radius: 10px;
        padding: 10px;
      }
      .rsfr-panel-head {
        display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px;
        font-size: 12px; color: var(--fg-dim, #7a8595); flex-wrap: wrap;
      }
      .rsfr-title { font-weight: 700; color: var(--fg, #e6edf3); font-size: 13px; }
      .rsfr-canvas-wrap {
        position: relative; background: #000; border-radius: 6px; overflow: hidden;
        width: 100%; max-height: 70vh;
      }
      .rsfr-canvas-wrap canvas {
        display: block; width: 100%; height: auto; max-height: 70vh; object-fit: contain;
      }
      .rsfr-loading {
        padding: 24px; text-align: center; color: rgba(255,255,255,0.6); font-size: 13px;
      }
      .rsfr-note {
        margin-top: 8px; padding: 8px 10px; border-radius: 8px;
        background: rgba(255,255,255,0.04); font-size: 12px;
        color: var(--fg-dim, #9aa4b2); line-height: 1.5;
      }
      .rsfr-note code {
        background: rgba(255,255,255,0.08); padding: 1px 5px; border-radius: 4px;
      }
      .rsfr-face {
        margin-top: 10px; padding: 10px; border-radius: 8px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
      }
      .rsfr-face-head {
        display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
        font-size: 13px;
      }
      .rsfr-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 3px 9px; border-radius: 999px; font-weight: 700;
        background: rgba(79,214,160,0.16); color: #4fd6a0;
        border: 1px solid rgba(79,214,160,0.4);
      }
      .rsfr-unknown {
        display: inline-flex; padding: 3px 9px; border-radius: 999px;
        background: rgba(255,170,80,0.12); color: #ffb85c;
        border: 1px solid rgba(255,170,80,0.35); font-weight: 600;
      }
      .rsfr-meta { color: var(--fg-dim, #9aa4b2); font-size: 12px; }
      .rsfr-votes {
        margin-top: 8px; display: flex; flex-direction: column; gap: 4px;
      }
      .rsfr-vote {
        display: flex; align-items: center; gap: 8px; font-size: 12px;
        font-family: ui-monospace, SFMono-Regular, monospace;
      }
      .rsfr-vote .rsfr-src { font-weight: 700; min-width: 64px; }
      .rsfr-status {
        padding: 1px 7px; border-radius: 5px; font-size: 11px;
        background: rgba(255,255,255,0.07); color: var(--fg-dim, #c2c8d2);
      }
      .rsfr-status.match { background: rgba(79,214,160,0.15); color: #4fd6a0; }
      .rsfr-status.below_threshold { background: rgba(255,170,80,0.14); color: #ffb85c; }
      .rsfr-status.ambiguous { background: rgba(255,120,120,0.14); color: #ff8a8a; }
      #tb-recognize[aria-pressed="true"] {
        color: #33d17a; border-color: rgba(51,209,122,0.55);
      }
    `;
    const tag = document.createElement('style');
    tag.id = 'rsfr-styles';
    tag.textContent = css;
    document.head.appendChild(tag);
  }

  function currentJobId() {
    // 与 compare.js 保持一致的兜底:`window.state.currentJobId` 没有时
    // 从 #pv-id 文本里抽 hex-dash-id。app.js 把 state 收紧在 IIFE 里,
    // 所以此处只能从 DOM 兜底 — 跟 compare.js 的实现完全一样。
    if (typeof window !== 'undefined' && window.state && window.state.currentJobId) return window.state.currentJobId;
    const idEl = document.getElementById('pv-id');
    if (idEl && idEl.textContent) {
      const m = idEl.textContent.match(/[#]?([0-9a-f-]+)/i);
      if (m) return m[1];
    }
    return null;
  }

  function removePanel() {
    const old = document.getElementById('rsfr-panel');
    if (old) old.remove();
  }

  function schedule() {
    if (state.pollingStopped) return;
    const detail = document.getElementById('pv-detail');
    if (!isEnabled() || !detail || detail.classList.contains('hidden')) {
      state.lastJobId = null;
      state.lastResponseKey = '';
      removePanel();
      return;
    }
    const jobId = currentJobId();
    if (!jobId) { state.lastJobId = null; return; }
    if (jobId === state.lastJobId) return;
    state.lastJobId = jobId;
    state.lastResponseKey = ''; // job 切换 → 旧 key 作废
    fetchAndRender(jobId);
  }

  function syncPolling() {
    const want = isEnabled() && !state.pollingStopped && document.visibilityState !== 'hidden';
    if (want && state.pollTimer === null) {
      state.pollTimer = setInterval(schedule, 600);
    } else if (!want && state.pollTimer !== null) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }
  document.addEventListener('visibilitychange', syncPolling);

  async function fetchAndRender(jobId) {
    const host = document.getElementById('pv-stage') || document.getElementById('pv-detail');
    if (!host) return;
    removePanel();
    const panel = document.createElement('div');
    panel.id = 'rsfr-panel';
    panel.className = 'rsfr-panel';
    panel.innerHTML = '<div class="rsfr-loading">Running LBPH / Eigenface / Fisherface consensus...</div>';
    host.appendChild(panel);
    try {
      // 后端要求 ?algos=haar(handler 用 q.algos 第一个切片决定 detector),
      // 旧的 ?detector=haar 文档是错的;统一走 ?algos= 路径。
      const resp = await fetch('/api/jobs/' + encodeURIComponent(jobId) + '/recognize?algos=haar', { method: 'POST' });
      const data = await resp.json();
      if (!resp.ok) {
        renderError(panel, resp.status, data);
        maybeStopPollingOnTerminalError(data);
        return;
      }
      // 响应去重:同 (face_count, 每个 face 的几何 + 共识标签) 直接返回,
      // 跳过重绘 canvas + 重建 ~50 个 DOM 节点。600ms tick × N 浏览器 tab
      // 是真实负载,这步把响应处理从 ~10ms 降到 ~0.1ms。
      const key = signatureOf(data);
      if (key === state.lastResponseKey) return;
      state.lastResponseKey = key;
      state.lastErrorKey = '';
      renderPanel(panel, data);
    } catch (e) {
      panel.innerHTML = '<div class="rsfr-loading" style="color:#ff8a8a">recognize failed: ' + (e.message || e) + '</div>';
    }
  }

  // 把响应简化成 cheap fingerprint,纯字符串拼接,比 JSON.stringify 快 ~10x。
  function signatureOf(data) {
    const faces = data.faces || [];
    if (!faces.length) return 'n=' + (data.face_count || 0) + '|' + (data.gallery_identities || 0);
    let s = 'n=' + faces.length + '|gi=' + (data.gallery_identities || 0);
    for (const f of faces) {
      const c = f.consensus;
      s += '|' + f.x + ',' + f.y + ',' + f.w + ',' + f.h + '|' + (c ? c.label + '@' + c.votes + '/' + c.confidence.toFixed(2) : '?');
    }
    return s;
  }

  // 终态错误(no_gallery / no_such_job / not_image_job / media_too_large):
  // 后端缓存已生效 / 不会自愈 → 停掉 setInterval,避免每 600ms 一次无效请求。
  // 用户点 toggle 按钮时 `initButton` 重置 pollingStopped。
  function maybeStopPollingOnTerminalError(data) {
    const code = data && data.error_code;
    const terminal = code === 'no_gallery' || code === 'no_such_job'
      || code === 'not_image_job' || code === 'media_too_large';
    if (terminal) {
      state.pollingStopped = true;
      syncPolling();
    }
  }

  function renderError(panel, status, data) {
    const code = data && data.error_code ? data.error_code : status;
    const msg = data && data.error ? data.error : ('HTTP ' + status);
    const errKey = String(code) + ':' + String(msg);
    if (errKey === state.lastErrorKey) return;
    state.lastErrorKey = errKey;
    let html = '<div class="rsfr-loading" style="color:#ff8a8a">[' + code + '] ' + msg + '</div>';
    if (code === 'no_gallery') {
      html = '<div class="rsfr-loading" style="color:#ffb85c">' + msg + '</div>' +
        '<div class="rsfr-note">Set <code>RSFACE_GALLERY_DIR</code> to a folder of identity folders ' +
        '(one subdirectory per person containing their .pgm/.ppm/.png crops), then restart the server.</div>';
    } else if (data && data.error_hint) {
      html += '<div class="rsfr-note">' + data.error_hint + '</div>';
    }
    panel.innerHTML = html;
  }

  function renderPanel(panel, data) {
    panel.innerHTML = '';

    const head = document.createElement('div');
    head.className = 'rsfr-panel-head';
    head.innerHTML =
      '<span class="rsfr-title">Identity consensus</span>' +
      '<span>gallery: ' + (data.gallery_identities || 0) + ' people / ' + (data.gallery_crops || 0) + ' crops</span>' +
      '<span>' + (data.width || '?') + 'x' + (data.height || '?') + ' / ' + (data.face_count || 0) + ' face(s)</span>';
    panel.appendChild(head);

    const wrap = document.createElement('div');
    wrap.className = 'rsfr-canvas-wrap';
    const canvas = document.createElement('canvas');
    wrap.appendChild(canvas);
    panel.appendChild(wrap);

    const faces = data.faces || [];
    drawOverlay(canvas, data.width, data.height, faces);

    if (!faces.length) {
      const note = document.createElement('div');
      note.className = 'rsfr-note';
      note.textContent = 'No face detected on this image, so there is nothing to identify.';
      panel.appendChild(note);
      return;
    }

    faces.forEach((face, index) => {
      panel.appendChild(renderFace(face, index + 1));
    });
  }

  function drawOverlay(canvas, width, height, faces) {
    const orig = document.getElementById('pv-img');
    const cw = width || (orig && orig.naturalWidth) || canvas.clientWidth || 640;
    const ch = height || (orig && orig.naturalHeight) || canvas.clientHeight || 480;
    canvas.width = cw;
    canvas.height = ch;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, cw, ch);
    ctx.lineWidth = Math.max(2, cw / 400);
    ctx.font = 'bold ' + Math.max(13, cw / 45) + 'px ui-monospace, monospace';
    ctx.textBaseline = 'top';

    faces.forEach((face) => {
      const known = !!face.consensus;
      const label = known ? face.consensus.label : 'unknown';
      const color = known ? '#4fd6a0' : '#ffb85c';
      ctx.strokeStyle = color;
      ctx.strokeRect(face.x, face.y, face.w, face.h);
      const text = label + (known ? '  ' + face.consensus.votes + 'v ' + Math.round(face.consensus.confidence * 100) + '%' : '');
      const metrics = ctx.measureText(text);
      const pad = 4;
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(face.x, Math.max(0, face.y - (cw / 45) - pad * 2), metrics.width + pad * 2, (cw / 45) + pad * 2);
      ctx.fillStyle = color;
      ctx.fillText(text, face.x + pad, Math.max(pad, face.y - (cw / 45) - pad));
    });
  }

  function renderFace(face, ordinal) {
    const box = document.createElement('div');
    box.className = 'rsfr-face';

    const headEl = document.createElement('div');
    headEl.className = 'rsfr-face-head';

    const badge = document.createElement('span');
    if (face.consensus) {
      badge.className = 'rsfr-badge';
      badge.textContent = '#' + ordinal + '  ' + face.consensus.label;
    } else {
      badge.className = 'rsfr-unknown';
      badge.textContent = '#' + ordinal + '  unknown / no consensus';
    }
    headEl.appendChild(badge);

    const boxInfo = document.createElement('span');
    boxInfo.className = 'rsfr-meta';
    boxInfo.textContent = '(' + face.x + ',' + face.y + ' ' + face.w + 'x' + face.h + ')';
    headEl.appendChild(boxInfo);

    if (face.consensus) {
      const info = document.createElement('span');
      info.className = 'rsfr-meta';
      info.textContent = face.consensus.votes + ' votes, conf ' +
        Math.round(face.consensus.confidence * 100) + '% [' + face.consensus.sources + ']';
      headEl.appendChild(info);
    }
    box.appendChild(headEl);

    const votes = document.createElement('div');
    votes.className = 'rsfr-votes';
    (face.recognizers || []).forEach((vote) => {
      const row = document.createElement('div');
      row.className = 'rsfr-vote';
      const meta = recognizerMeta(vote.source);
      const distText = (vote.distance === null || vote.distance === undefined) ? '' : ' d=' + Number(vote.distance).toFixed(3);
      row.innerHTML =
        '<span class="rsfr-src" style="color:' + meta.color + '">' + meta.label + '</span>' +
        '<span class="rsfr-status ' + vote.status + '">' + vote.status.replace('_', ' ') + '</span>' +
        '<span class="rsfr-meta">' + (vote.label ? vote.label : '—') + distText + '</span>';
      votes.appendChild(row);
    });
    box.appendChild(votes);
    return box;
  }

  function initButton() {
    const btn = document.getElementById('tb-recognize');
    if (!btn) return;
    injectStyles();
    btn.setAttribute('aria-pressed', isEnabled() ? 'true' : 'false');
    btn.addEventListener('click', () => {
      const next = !isEnabled();
      setEnabled(next);
      btn.setAttribute('aria-pressed', next ? 'true' : 'false');
      // 用户切换:重新允许轮询(上次可能是 no_gallery 自动停的)。
      state.pollingStopped = false;
      state.lastResponseKey = '';
      state.lastErrorKey = '';
      removePanel();
      syncPolling();
      schedule();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      initButton();
      syncPolling();
    });
  } else {
    initButton();
    syncPolling();
  }
})();
