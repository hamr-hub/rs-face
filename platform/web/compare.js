/* rs-face / platform/web/compare.js
 *
 * Algorithm compare mode (multi-algo overlay).
 *
 * Loaded via `index.html` (defer, before app.js).
 *
 * What it does:
 *  1. The topbar button (id=`tb-compare`) toggles algorithm compare mode.
 *     State is persisted to localStorage.
 *  2. When on, the module fetches
 *       POST /api/jobs/{id}/compare?algos=haar,cnn,luminance
 *     for the currently-open job. (Server returns bbox list per algo.)
 *  3. The module renders one large overlay canvas with ALL algorithms'
 *     detections drawn on top of the source image, each in its own color
 *     (Haar=red, CNN=green, Luminance=cyan, SCRFD=blue, Ensemble=gold).
 *  4. A legend below the canvas shows each algorithm's bbox count, avg
 *     confidence, and a toggle (click to enable/disable drawing that algo).
 *  5. Each bbox is labeled with the algorithm abbreviation and confidence
 *     score (e.g. "H 0.82", "C 0.55"). Hidden algorithms are skipped.
 *  6. Toggling off removes the panel.
 *
 * Zero new JS dependencies. Only DOM + canvas. Talks to existing /api/*
 * endpoints. Injects all CSS / DOM at runtime (no index.html / style.css
 * edits required for the panel itself).
 */
(() => {
  'use strict';

  const STORAGE_KEY = 'rsface.compare.enabled';

  // Algo meta. SCRFD / Ensemble are placeholders — backend currently exposes
  // haar / cnn / luminance; the overlay UI is designed to render any extra
  // algos the server adds in the future without code changes.
  const ALGOS_DEFAULT = ['haar', 'cnn', 'luminance'];
  const ALGO_META = {
    haar:      { abbr: 'H', label: 'HAAR',      color: [0, 255, 96],   desc: 'Viola-Jones Haar cascade (2001)' },
    cnn:       { abbr: 'C', label: 'CNN',       color: [80, 220, 120], desc: 'small CNN, 24x24 Conv+ReLU+FC' },
    luminance: { abbr: 'L', label: 'LUM',       color: [80, 220, 220], desc: 'Luminance bands + mirror symmetry + edge density (no weights)' },
    scrfd:     { abbr: 'S', label: 'SCRFD',     color: [120, 160, 255], desc: 'SCRFD face detector (planned)' },
    ensemble:  { abbr: 'E', label: 'ENSEMBLE',  color: [255, 215, 0],  desc: 'Ensemble fuser across algorithms' },
  };

  // UI state.
  const state = {
    enabled: false,
    visibility: {},   // algo -> bool, default all true
    lastJobId: null,
    lastResults: null, // remember results so toggles re-render without re-fetch
    lastOrigSrc: null,
  };

  function isEnabled() {
    try { return localStorage.getItem(STORAGE_KEY) === '1'; } catch { return false; }
  }
  function setEnabled(v) {
    state.enabled = !!v;
    try { localStorage.setItem(STORAGE_KEY, v ? '1' : '0'); } catch {}
  }

  function algoMeta(name) {
    const meta = ALGO_META[name];
    if (meta) return meta;
    // unknown algo — derive an abbreviation + deterministic color from name
    const abbr = String(name).charAt(0).toUpperCase();
    let hash = 0; for (const c of String(name)) hash = (hash * 31 + c.charCodeAt(0)) & 0xff;
    return { abbr, label: String(name).toUpperCase(), color: [hash, 200 - (hash % 80), 255 - (hash % 80)], desc: name };
  }

  function injectStyles() {
    if (document.getElementById('rsfc-styles')) return;
    const css = `
      .rsfc-menu {
        position: fixed; z-index: 9999;
        background: var(--bg, #1a1c22); color: var(--fg, #e7e8ea);
        border: 1px solid rgba(255,255,255,0.12); border-radius: 10px;
        padding: 10px 12px; min-width: 240px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.45);
        font-size: 13px;
      }
      .rsfc-menu.hidden { display: none; }
      .rsfc-row { display: flex; align-items: center; gap: 8px; padding: 6px 0; }
      .rsfc-row .rsfc-label { font-weight: 600; }
      .rsfc-row .rsfc-hint { color: rgba(255,255,255,0.55); font-size: 12px; }
      .rsfc-info { flex-direction: column; align-items: flex-start; gap: 4px; }

      .rsfc-panel {
        margin: 12px 0;
        background: var(--bg-3, #181d27);
        border: 1px solid var(--border, rgba(255,255,255,0.08));
        border-radius: 10px;
        padding: 10px;
      }
      .rsfc-panel-head {
        display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px;
        font-size: 12px; color: var(--fg-dim, #7a8595);
      }
      .rsfc-panel-head .rsfc-title { font-weight: 700; color: var(--fg, #e6edf3); font-size: 13px; }
      .rsfc-canvas-wrap {
        position: relative; background: #000; border-radius: 6px; overflow: hidden;
        width: 100%; max-height: 70vh;
      }
      .rsfc-canvas-wrap canvas {
        display: block; width: 100%; height: auto; max-height: 70vh; object-fit: contain;
      }
      .rsfc-err-overlay {
        position: absolute; inset: 0;
        display: flex; align-items: center; justify-content: center;
        background: rgba(0,0,0,0.5); color: #ff7070; padding: 12px; text-align: center;
        font-size: 12px;
      }
      .rsfc-legend {
        display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;
        font-family: ui-monospace, SFMono-Regular, monospace; font-size: 11px;
      }
      .rsfc-chip {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 8px; border-radius: 6px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        cursor: pointer; user-select: none;
        color: var(--fg-dim, #7a8595);
        transition: background-color .15s, color .15s, opacity .15s;
      }
      .rsfc-chip:hover { background: rgba(255,255,255,0.08); }
      .rsfc-chip[aria-pressed="false"] {
        opacity: 0.45; text-decoration: line-through;
      }
      .rsfc-chip .rsfc-swatch {
        display: inline-block; width: 10px; height: 10px; border-radius: 2px;
        box-shadow: 0 0 0 1px rgba(0,0,0,0.3) inset;
      }
      .rsfc-chip .rsfc-abbr { font-weight: 700; }
      .rsfc-chip .rsfc-count { color: rgba(255,255,255,0.55); }
      .rsfc-chip .rsfc-conf { color: rgba(255,255,255,0.65); }
      .rsfc-loading {
        padding: 24px; text-align: center; color: rgba(255,255,255,0.6);
        font-size: 13px;
      }
      #tb-compare[aria-pressed="true"] {
        color: #33d17a;
        border-color: rgba(51,209,122,0.55);
      }
    `;
    const tag = document.createElement('style');
    tag.id = 'rsfc-styles';
    tag.textContent = css;
    document.head.appendChild(tag);
  }

  function ensureSettingsMenu() {
    let menu = document.getElementById('rsfc-settings-menu');
    if (menu) return menu;
    const btn = document.getElementById('tb-compare');
    if (!btn) return null;
    injectStyles();
    menu = document.createElement('div');
    menu.id = 'rsfc-settings-menu';
    menu.className = 'rsfc-menu hidden';
    menu.innerHTML = `
      <label class="rsfc-row">
        <input type="checkbox" id="rsfc-cmp-toggle" ${isEnabled() ? 'checked' : ''}>
        <span class="rsfc-label">Algorithm compare mode</span>
        <span class="rsfc-hint">multi-algo overlay</span>
      </label>
      <div class="rsfc-row rsfc-info">
        <span class="rsfc-hint">When on, image jobs render one canvas with all algorithms' detections overlaid. Click legend chips to toggle each algo.</span>
      </div>
    `;
    document.body.appendChild(menu);
    btn.setAttribute('aria-pressed', isEnabled() ? 'true' : 'false');
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const r = btn.getBoundingClientRect();
      menu.style.top = (r.bottom + 6) + 'px';
      menu.style.right = (window.innerWidth - r.right) + 'px';
      menu.classList.toggle('hidden');
    });
    document.addEventListener('click', () => menu.classList.add('hidden'));
    menu.addEventListener('click', (e) => e.stopPropagation());
    const toggle = menu.querySelector('#rsfc-cmp-toggle');
    toggle.addEventListener('change', () => {
      setEnabled(toggle.checked);
      btn.setAttribute('aria-pressed', toggle.checked ? 'true' : 'false');
      syncPolling();
      scheduleCompare();
    });
    return menu;
  }

  function currentJobId() {
    if (typeof state !== 'undefined' && window.state && window.state.currentJobId) return window.state.currentJobId;
    const idEl = document.getElementById('pv-id');
    if (idEl && idEl.textContent) {
      const m = idEl.textContent.match(/[#]?([0-9a-f-]+)/i);
      if (m) return m[1];
    }
    return null;
  }

  function scheduleCompare() {
    const enabled = isEnabled();
    const detail = document.getElementById('pv-detail');
    if (!enabled || !detail || detail.classList.contains('hidden')) {
      state.lastJobId = null;
      removeComparePanel();
      return;
    }
    const jobId = currentJobId();
    if (!jobId) { state.lastJobId = null; return; }
    if (jobId === state.lastJobId) return;
    state.lastJobId = jobId;
    fetchAndRenderCompare(jobId);
  }

  let pollTimer = null;
  function syncPolling() {
    const want = isEnabled() && document.visibilityState !== 'hidden';
    if (want && pollTimer === null) {
      pollTimer = setInterval(scheduleCompare, 600);
    } else if (!want && pollTimer !== null) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }
  document.addEventListener('visibilitychange', syncPolling);

  async function fetchAndRenderCompare(jobId) {
    const host = document.getElementById('pv-stage') || document.getElementById('pv-detail');
    if (!host) return;
    removeComparePanel();
    const panel = document.createElement('div');
    panel.id = 'rsfc-compare-panel';
    panel.className = 'rsfc-panel';
    panel.innerHTML = '<div class="rsfc-loading">Running ' + ALGOS_DEFAULT.length + ' algos in parallel (haar/cnn/luminance)...</div>';
    host.appendChild(panel);
    try {
      const resp = await fetch('/api/jobs/' + encodeURIComponent(jobId) + '/compare?algos=' + ALGOS_DEFAULT.join(','), { method: 'POST' });
      if (!resp.ok) {
        panel.innerHTML = '<div class="rsfc-loading" style="color:#ff7070">compare failed: HTTP ' + resp.status + '</div>';
        return;
      }
      const data = await resp.json();
      const orig = document.getElementById('pv-img');
      const origSrc = orig && orig.src;
      state.lastResults = data;
      state.lastOrigSrc = origSrc;
      renderComparePanel(panel, data, origSrc);
    } catch (e) {
      panel.innerHTML = '<div class="rsfc-loading" style="color:#ff7070">compare failed: ' + (e.message || e) + '</div>';
    }
  }

  function removeComparePanel() {
    const old = document.getElementById('rsfc-compare-panel');
    if (old) old.remove();
  }

  function renderComparePanel(panel, data, origSrc) {
    panel.innerHTML = '';
    const results = data.results || [];
    // Build per-algo visibility defaults (preserve user's toggles from prior runs).
    for (const r of results) {
      const name = r.algo;
      if (state.visibility[name] === undefined) state.visibility[name] = true;
    }

    // ---- head ----
    const head = document.createElement('div');
    head.className = 'rsfc-panel-head';
    const headTitle = document.createElement('span');
    headTitle.className = 'rsfc-title';
    headTitle.textContent = 'Algorithm compare (overlay)';
    head.appendChild(headTitle);
    const headMeta = document.createElement('span');
    headMeta.textContent = (data.width || '?') + 'x' + (data.height || '?') + ' / ' + results.length + ' algos';
    head.appendChild(headMeta);
    panel.appendChild(head);

    // ---- canvas ----
    const wrap = document.createElement('div');
    wrap.className = 'rsfc-canvas-wrap';
    const canvas = document.createElement('canvas');
    wrap.appendChild(canvas);

    // surface per-algo errors on the canvas (overlay, no error per-card now)
    const errs = results.filter(r => r.error);
    if (errs.length === results.length && results.length > 0) {
      const errEl = document.createElement('div');
      errEl.className = 'rsfc-err-overlay';
      errEl.textContent = 'All algorithms failed: ' + errs.map(e => e.algo + ': ' + e.error).join(' · ');
      wrap.appendChild(errEl);
    } else if (errs.length) {
      const errEl = document.createElement('div');
      errEl.className = 'rsfc-err-overlay';
      errEl.style.alignItems = 'flex-start';
      errEl.style.justifyContent = 'flex-end';
      errEl.style.padding = '6px 10px';
      errEl.style.fontSize = '11px';
      errEl.textContent = errs.map(e => e.algo + ': ' + e.error).join(' · ');
      wrap.appendChild(errEl);
    }
    panel.appendChild(wrap);

    // ---- legend ----
    const legend = document.createElement('div');
    legend.className = 'rsfc-legend';
    legend.setAttribute('role', 'group');
    legend.setAttribute('aria-label', 'Algorithm legend');
    for (const r of results) {
      legend.appendChild(makeChip(r));
    }
    panel.appendChild(legend);

    // ---- load image, draw everything ----
    drawOverlay(canvas, wrap, data, results, origSrc);
  }

  function makeChip(result) {
    const meta = algoMeta(result.algo);
    const vis = state.visibility[result.algo] !== false;
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'rsfc-chip';
    chip.dataset.algo = result.algo;
    chip.setAttribute('aria-pressed', vis ? 'true' : 'false');
    chip.title = meta.desc + (result.error ? ' — error: ' + result.error : '');
    const swatch = document.createElement('span');
    swatch.className = 'rsfc-swatch';
    swatch.style.background = 'rgb(' + meta.color.join(',') + ')';
    chip.appendChild(swatch);
    const abbr = document.createElement('span');
    abbr.className = 'rsfc-abbr';
    abbr.textContent = meta.label;
    chip.appendChild(abbr);
    const cnt = document.createElement('span');
    cnt.className = 'rsfc-count';
    cnt.textContent = (result.detection_count != null ? result.detection_count : (result.detections || []).length) + ' bbox';
    chip.appendChild(cnt);
    // avg confidence
    const dets = result.detections || [];
    let avg = null;
    if (dets.length) {
      let sum = 0, n = 0;
      for (const d of dets) if (typeof d.score === 'number') { sum += d.score; n++; }
      if (n > 0) avg = sum / n;
    }
    if (avg != null) {
      const conf = document.createElement('span');
      conf.className = 'rsfc-conf';
      conf.textContent = 'avg ' + avg.toFixed(2);
      chip.appendChild(conf);
    }
    if (result.elapsed_ms != null) {
      const ms = document.createElement('span');
      ms.className = 'rsfc-conf';
      ms.textContent = result.elapsed_ms + ' ms';
      chip.appendChild(ms);
    }
    chip.addEventListener('click', () => {
      const newVal = !(state.visibility[result.algo] !== false);
      state.visibility[result.algo] = newVal;
      chip.setAttribute('aria-pressed', newVal ? 'true' : 'false');
      // re-draw without re-fetching
      redrawFromCache();
    });
    return chip;
  }

  function redrawFromCache() {
    const panel = document.getElementById('rsfc-compare-panel');
    if (!panel || !state.lastResults) return;
    const wrap = panel.querySelector('.rsfc-canvas-wrap');
    const canvas = wrap && wrap.querySelector('canvas');
    if (!canvas) return;
    drawOverlay(canvas, wrap, state.lastResults, state.lastResults.results || [], state.lastOrigSrc);
  }

  function drawOverlay(canvas, wrap, data, results, origSrc) {
    if (!origSrc) {
      // No source image yet (preview stage not loaded) — degrade gracefully.
      const w = data.width || 320, h = data.height || 240;
      canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#0e1014';
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.font = '13px sans-serif';
      ctx.fillText('原图尚未加载,等待预览就绪后会自动叠加', 12, 24);
      return;
    }
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const w = img.naturalWidth || data.width || 320;
      const h = img.naturalHeight || data.height || 240;
      canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);

      // Compute label size from image dimensions (long-edge normalized).
      const fontPx = Math.max(10, Math.round(w / 38));
      ctx.font = 'bold ' + fontPx + 'px ui-monospace, SFMono-Regular, monospace';
      ctx.textBaseline = 'top';

      for (const r of results) {
        if (state.visibility[r.algo] === false) continue;
        const meta = algoMeta(r.algo);
        const color = 'rgb(' + meta.color.join(',') + ')';
        ctx.lineWidth = Math.max(1.5, w / 240);
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        for (const d of (r.detections || [])) {
          ctx.strokeRect(d.x, d.y, d.w, d.h);
          if (d.score !== undefined) {
            const label = meta.abbr + ' ' + d.score.toFixed(2);
            // label background for readability
            const tw = ctx.measureText(label).width;
            const pad = 3;
            const lx = Math.max(0, d.x);
            const ly = Math.max(0, d.y - fontPx - pad * 2);
            ctx.save();
            ctx.fillStyle = 'rgba(0,0,0,0.55)';
            ctx.fillRect(lx, ly, tw + pad * 2, fontPx + pad * 2);
            ctx.fillStyle = color;
            ctx.fillText(label, lx + pad, ly + pad);
            ctx.restore();
          }
        }
      }
    };
    img.onerror = () => {
      const ctx = canvas.getContext('2d');
      canvas.width = 320; canvas.height = 240;
      ctx.fillStyle = '#0e1014';
      ctx.fillRect(0, 0, 320, 240);
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.font = '13px sans-serif';
      ctx.fillText('原图加载失败,无法叠加', 12, 24);
    };
    img.src = origSrc;
  }

  function init() {
    ensureSettingsMenu();
    syncPolling();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();