/* rs-face / platform/web/timeline.js
 *
 * Per-frame job timeline. Renders one dot per frame, colored by status:
 *   - green: frame has at least one face detection
 *   - gray:  frame processed but no faces
 *   - red:   frame errored (rare; visible on bad streams)
 *   - amber: frame queued but not yet processed
 *
 * X axis: time (0..max timestamp_ms).
 * Y axis: frame index (0..N-1).
 *
 * Interactions:
 *   - hover a dot -> tooltip with frame number / timestamp / face count / GPU level
 *   - click a dot -> open the frame detail modal (preview.jumpToFrame) if available,
 *     otherwise no-op (graceful: feature must not crash when modal isn't ready).
 *
 * Zero new JS dependencies. Pure DOM + Canvas + Tooltip div.
 *
 * Loaded via `index.html` (defer, after app.js so that `state`, `utils`, `preview`,
 * and `app` global symbols are available).
 */
(() => {
  'use strict';

  // Color tokens (read once at render time so dark/light switch works without re-init).
  function colors() {
    const cs = getComputedStyle(document.documentElement);
    return {
      face:    cs.getPropertyValue('--success').trim() || '#3ddc84',
      noFace:  cs.getPropertyValue('--fg-dim-2').trim() || '#6f7f96',
      error:   cs.getPropertyValue('--danger').trim()  || '#f85149',
      pending: cs.getPropertyValue('--warn').trim()    || '#f0c674',
      axis:    cs.getPropertyValue('--border-2').trim() || 'rgba(255,255,255,0.12)',
      axisFg:  cs.getPropertyValue('--fg-dim').trim() || '#7a8595',
      bg:      cs.getPropertyValue('--bg-3').trim()    || '#181d27',
      bgCard:  cs.getPropertyValue('--bg-2').trim()    || '#11151d',
      fg:      cs.getPropertyValue('--fg').trim()      || '#e6edf3',
      fgDim:   cs.getPropertyValue('--fg-dim').trim()  || '#7a8595',
    };
  }

  let _initialized = false;
  let _host = null;
  let _canvas = null;
  let _tooltip = null;
  let _caption = null;
  let _hint = null;
  let _frames = null;   // last rendered frames array (for hover hit-test)
  let _tStart = 0;
  let _tEnd = 1;
  let _jobId = null;
  let _frameMaxIdx = 0;
  let _pending = false;

  function ensureDom() {
    if (_initialized) return;
    _initialized = true;
    _host = document.getElementById('timeline');
    if (!_host) return;
    injectStyles();
    _host.innerHTML = `
      <div class="tl-head">
        <span class="tl-title">帧时间轴</span>
        <span class="tl-caption" id="timeline-caption">无数据</span>
        <span class="tl-hint">悬停查看详情 · 点击跳转</span>
      </div>
      <div class="tl-stage">
        <canvas id="timeline-canvas" aria-label="帧时间轴" role="img"></canvas>
        <div class="tl-tooltip hidden" id="timeline-tooltip" role="tooltip"></div>
      </div>
      <div class="tl-legend" aria-label="颜色图例">
        <span class="tl-leg"><span class="tl-dot tl-face"></span>有人脸</span>
        <span class="tl-leg"><span class="tl-dot tl-noface"></span>无脸</span>
        <span class="tl-leg"><span class="tl-dot tl-pending"></span>未处理</span>
        <span class="tl-leg"><span class="tl-dot tl-error"></span>错误</span>
      </div>
    `;
    _canvas = document.getElementById('timeline-canvas');
    _tooltip = document.getElementById('timeline-tooltip');
    _caption = document.getElementById('timeline-caption');
    _hint = _host.querySelector('.tl-hint');

    // Hit-test handlers on the canvas.
    _canvas.addEventListener('mousemove', onMove);
    _canvas.addEventListener('mouseleave', hideTooltip);
    _canvas.addEventListener('click', onClick);

    // Re-render on theme change (color tokens depend on --success / --danger / etc.)
    if (window.matchMedia) {
      const mq = window.matchMedia('(prefers-color-scheme: light)');
      mq.addEventListener('change', () => { if (_frames) render(_frames, _jobId); });
    }
  }

  function injectStyles() {
    if (document.getElementById('tl-styles')) return;
    const css = `
      #timeline {
        background: var(--bg-2, #11151d);
        border: 1px solid var(--border, rgba(255,255,255,0.08));
        border-radius: 8px;
        padding: 10px 12px;
        margin: 8px 0 4px 0;
        font-family: ui-monospace, SFMono-Regular, monospace;
        font-size: 12px;
        color: var(--fg, #e6edf3);
      }
      #timeline .tl-head {
        display: flex; align-items: baseline; gap: 12px; margin-bottom: 6px;
      }
      #timeline .tl-title { font-weight: 700; font-size: 12px; }
      #timeline .tl-caption { color: var(--fg-dim, #7a8595); font-size: 11px; flex: 1; }
      #timeline .tl-hint { color: var(--fg-dim-2, #6f7f96); font-size: 10px; }
      #timeline .tl-stage {
        position: relative;
        background: var(--bg-3, #181d27);
        border: 1px solid var(--border, rgba(255,255,255,0.06));
        border-radius: 6px;
        width: 100%;
        overflow: hidden;
      }
      #timeline canvas {
        display: block; width: 100%; height: 110px;
        cursor: crosshair;
      }
      #timeline .tl-tooltip {
        position: absolute; pointer-events: none;
        background: rgba(20, 24, 32, 0.96);
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 6px;
        padding: 6px 8px;
        font-size: 11px;
        line-height: 1.45;
        color: #e6edf3;
        white-space: pre;
        box-shadow: 0 6px 18px rgba(0,0,0,0.4);
        z-index: 5;
      }
      #timeline .tl-tooltip.hidden { display: none; }
      #timeline .tl-legend {
        display: flex; flex-wrap: wrap; gap: 12px; margin-top: 6px;
        font-size: 10px; color: var(--fg-dim, #7a8595);
      }
      #timeline .tl-leg { display: inline-flex; align-items: center; gap: 4px; }
      #timeline .tl-dot {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
      }
      #timeline .tl-dot.tl-face { background: var(--success, #3ddc84); }
      #timeline .tl-dot.tl-noface { background: var(--fg-dim-2, #6f7f96); }
      #timeline .tl-dot.tl-pending { background: var(--warn, #f0c674); }
      #timeline .tl-dot.tl-error { background: var(--danger, #f85149); }
    `;
    const tag = document.createElement('style');
    tag.id = 'tl-styles';
    tag.textContent = css;
    document.head.appendChild(tag);
  }

  /**
   * Public API: timeline.render(frames, jobId)
   * @param {Array} frames - [{ index, timestamp_ms, faces, status, gpu_level? }]
   * @param {string} jobId  - current job id (for click routing)
   */
  function render(frames, jobId) {
    ensureDom();
    if (!_host) return;
    _frames = Array.isArray(frames) ? frames.slice() : [];
    _jobId = jobId || (window.state && window.state.currentJobId) || null;
    if (!_frames.length) {
      _caption.textContent = '无数据';
      _host.classList.add('hidden');
      drawEmpty();
      return;
    }
    _host.classList.remove('hidden');
    // Time domain: 0..max(timestamp_ms) — clamp >0 to avoid div-by-zero.
    let tMax = 0;
    let faceCount = 0;
    let noFaceCount = 0;
    let errCount = 0;
    let pendCount = 0;
    let lastIdx = 0;
    for (const f of _frames) {
      const t = +f.timestamp_ms || 0;
      if (t > tMax) tMax = t;
      const n = (f.faces || []).length;
      if (n > 0) faceCount++; else noFaceCount++;
      if (f.status === 'error' || f.error) errCount++;
      if (f.status === 'pending' || f.status === 'queued') pendCount++;
      if (typeof f.index === 'number' && f.index > lastIdx) lastIdx = f.index;
    }
    _tStart = 0;
    _tEnd = Math.max(1, tMax);
    _frameMaxIdx = Math.max(1, lastIdx);
    _caption.textContent = `${_frames.length} 帧 / 有人脸 ${faceCount} / 无脸 ${noFaceCount}` +
      (errCount ? ` / 错误 ${errCount}` : '') +
      (pendCount ? ` / 未处理 ${pendCount}` : '') +
      ` · ${(tMax / 1000).toFixed(1)}s`;

    // Force canvas size to match its CSS width so DPI is sharp.
    requestAnimationFrame(() => drawAll());
  }

  function drawEmpty() {
    if (!_canvas) return;
    const ctx = _canvas.getContext('2d');
    const w = _canvas.clientWidth || 400;
    const h = 110;
    const dpr = window.devicePixelRatio || 1;
    _canvas.width = w * dpr;
    _canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = colors().fgDim;
    ctx.font = '12px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('暂无帧数据', w / 2, h / 2);
    ctx.textAlign = 'start';
  }

  function drawAll() {
    if (!_canvas || !_frames) return;
    const dpr = window.devicePixelRatio || 1;
    const w = _canvas.clientWidth || 600;
    const h = _canvas.clientHeight || 110;
    _canvas.width = w * dpr;
    _canvas.height = h * dpr;
    const ctx = _canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const c = colors();
    ctx.clearRect(0, 0, w, h);

    const padL = 36, padR = 8, padT = 8, padB = 18;
    const plotW = Math.max(1, w - padL - padR);
    const plotH = Math.max(1, h - padT - padB);

    // Background of plot area.
    ctx.fillStyle = c.bg;
    ctx.fillRect(padL, padT, plotW, plotH);

    // Y axis ticks (frame index: 0, mid, max).
    ctx.strokeStyle = c.axis;
    ctx.fillStyle = c.axisFg;
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 1;
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const idx = Math.round((_frameMaxIdx * i) / yTicks);
      const y = padT + plotH - (i / yTicks) * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.globalAlpha = i === yTicks ? 0.4 : 0.12;
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillText('#' + idx, padL - 4, y);
    }
    // Y axis label.
    ctx.save();
    ctx.translate(10, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillStyle = c.fgDim;
    ctx.fillText('frame idx', 0, 0);
    ctx.restore();

    // X axis ticks (time).
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xTicks = 6;
    for (let i = 0; i <= xTicks; i++) {
      const t = (_tEnd * i) / xTicks;
      const x = padL + (i / xTicks) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, padT + plotH);
      ctx.lineTo(x, padT + plotH + 3);
      ctx.globalAlpha = 0.5;
      ctx.strokeStyle = c.axis;
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = c.axisFg;
      ctx.fillText((t / 1000).toFixed(1) + 's', x, padT + plotH + 5);
    }
    // X axis label.
    ctx.fillStyle = c.fgDim;
    ctx.textAlign = 'right';
    ctx.fillText('time', padL + plotW, h - 2);

    // Frame dots.
    const range = Math.max(1, _tEnd - _tStart);
    const frMax = Math.max(1, _frameMaxIdx);
    for (const f of _frames) {
      const t = +f.timestamp_ms || 0;
      const idx = typeof f.index === 'number' ? f.index : 0;
      const x = padL + ((t - _tStart) / range) * plotW;
      const y = padT + plotH - (idx / frMax) * plotH;
      const hasFaces = (f.faces || []).length > 0;
      const isError = f.status === 'error' || !!f.error;
      const isPending = f.status === 'pending' || f.status === 'queued';
      let fill;
      if (isError) fill = c.error;
      else if (isPending) fill = c.pending;
      else if (hasFaces) fill = c.face;
      else fill = c.noFace;
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }

    // Stash dots for hit-test (relative to canvas local coords).
    _dotMap = [];
    for (const f of _frames) {
      const t = +f.timestamp_ms || 0;
      const idx = typeof f.index === 'number' ? f.index : 0;
      const x = padL + ((t - _tStart) / range) * plotW;
      const y = padT + plotH - (idx / frMax) * plotH;
      _dotMap.push({ x, y, frame: f });
    }
  }

  // Hit-test cache (rebuilt on each draw).
  let _dotMap = [];

  function findDotAt(x, y) {
    // Find nearest dot within 6px (in canvas-local coordinates).
    let best = null, bestD = 6 * 6;
    for (const d of _dotMap) {
      const dx = d.x - x, dy = d.y - y;
      const dist2 = dx * dx + dy * dy;
      if (dist2 < bestD) { bestD = dist2; best = d; }
    }
    return best;
  }

  function onMove(e) {
    if (!_frames || !_frames.length || !_canvas) return;
    const rect = _canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const dpr = window.devicePixelRatio || 1;
    const dot = findDotAt(x * dpr / dpr, y); // coordinates already in CSS px
    if (!dot) { hideTooltip(); return; }
    showTooltip(e, dot);
  }

  function showTooltip(e, dot) {
    if (!_tooltip || !_frames) return;
    const f = dot.frame;
    const lines = [];
    lines.push(`#${f.index != null ? f.index : '?'}`);
    lines.push(`t = ${((+f.timestamp_ms || 0) / 1000).toFixed(2)} s`);
    const n = (f.faces || []).length;
    lines.push(`人脸 ${n}`);
    if (f.gpu_level != null) lines.push(`GPU ${f.gpu_level}`);
    if (f.status === 'error' || f.error) lines.push('ERROR ' + (f.error || ''));
    if (f.status === 'pending' || f.status === 'queued') lines.push('PENDING');
    _tooltip.textContent = lines.join('\n');
    _tooltip.classList.remove('hidden');
    const rect = _canvas.getBoundingClientRect();
    const stageRect = _tooltip.parentElement.getBoundingClientRect();
    const tx = e.clientX - stageRect.left + 10;
    const ty = e.clientY - stageRect.top + 10;
    // Keep tooltip inside the stage.
    const tipW = _tooltip.offsetWidth || 120;
    const tipH = _tooltip.offsetHeight || 60;
    const maxX = stageRect.width - tipW - 4;
    const maxY = stageRect.height - tipH - 4;
    _tooltip.style.left = Math.min(maxX, Math.max(0, tx)) + 'px';
    _tooltip.style.top  = Math.min(maxY, Math.max(0, ty)) + 'px';
    void rect; // unused, kept for future use
  }

  function hideTooltip() {
    if (_tooltip) _tooltip.classList.add('hidden');
  }

  function onClick(e) {
    if (!_frames || !_frames.length || !_canvas) return;
    const rect = _canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const dot = findDotAt(x, y);
    if (!dot) return;
    const f = dot.frame;
    // Try to open the existing frame lightbox (preview.showFrame if exposed).
    try {
      if (typeof window !== 'undefined' && window.app && typeof window.app.openFrame === 'function') {
        window.app.openFrame(_jobId, f.index != null ? f.index : 0);
        return;
      }
      if (window.preview && typeof window.preview.openFrame === 'function') {
        window.preview.openFrame(_jobId, f.index != null ? f.index : 0);
        return;
      }
      // Fallback: navigate via deep-link to frame.
      if (_jobId) {
        const idx = f.index != null ? f.index : 0;
        history.replaceState(null, '', `#/job/${encodeURIComponent(_jobId)}/frame/${idx}`);
      }
    } catch (err) {
      console.warn('[timeline] click handler failed:', err);
    }
  }

  /** Read frames from window.state.currentJob and render. Auto-pulled by app.js. */
  function refreshFromState() {
    if (!window.state || !window.state.currentJob) return;
    const job = window.state.currentJob;
    render(job.frames || [], job.id);
  }

  /** Hook into state.currentJob mutations: render whenever job changes. */
  let _lastJobId = null;
  function watchState() {
    if (_pending) return;
    _pending = true;
    const tick = () => {
      _pending = false;
      const s = window.state;
      if (!s || !s.currentJob) return;
      if (s.currentJob.id !== _lastJobId) {
        _lastJobId = s.currentJob.id;
        render(s.currentJob.frames || [], s.currentJob.id);
      } else if (s.currentJob.frames && s.currentJob.frames !== _frames) {
        // frames array may have been replaced (SSE update) — re-render.
        render(s.currentJob.frames || [], s.currentJob.id);
      }
    };
    const loop = () => {
      tick();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  function init() {
    ensureDom();
    watchState();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Expose API.
  window.timeline = { render, refreshFromState };
})();