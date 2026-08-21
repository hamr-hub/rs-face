/* rs-face / platform/web/compare.js
 *
 * Independent module for the "algorithm compare mode" feature entry.
 *
 * Load (the doubao agent will add this after their layout work):
 *     <script type="module" src="/compare.js"></script>
 *
 * What it does:
 *  1. Clicking the topbar gear button (id=`tb-settings`) opens a small
 *     dropdown menu containing an "algorithm compare mode" checkbox.
 *     The state is persisted to localStorage.
 *  2. When the toggle is on and a job enters preview, the module
 *     automatically calls
 *       POST /api/jobs/{id}/compare?algos=haar,cnn,yunet,mtcnn,hog
 *  3. The backend returns each algo's detection count + elapsed ms +
 *     boxes. The frontend re-uses the same source image 5 times,
 *     each in its own canvas, each algorithm's boxes drawn in its
 *     own colour, side by side.
 *  4. Toggling off removes the panel.
 *
 * Zero new dependencies. Only calls `/api/*` endpoints that the
 * platform already exposes. Does NOT modify index.html / style.css
 * (the doubao agent owns those files) - only operates on its own
 * `.rsfc-` prefixed DOM.
 */
(() => {
  'use strict';

  const STORAGE_KEY = 'rsface.compare.enabled';
  const ALGOS = ['haar', 'cnn', 'yunet', 'mtcnn', 'hog'];
  const ALGO_COLORS = {
    haar:  [0, 255, 96],
    cnn:   [255, 170, 0],
    yunet: [120, 200, 255],
    mtcnn: [220, 120, 255],
    hog:   [255, 80, 160],
  };
  const ALGO_DESC = {
    haar:  'Viola-Jones Haar cascade (2001)',
    cnn:   'small CNN, 24x24 Conv+ReLU+FC',
    yunet: 'YuNet-style anchor-based, 5 scales',
    mtcnn: 'MTCNN 3-stage cascade (P/R/O-Net)',
    hog:   'HOG 8x8 + Linear SVM, 64x128',
  };

  function isEnabled() {
    try { return localStorage.getItem(STORAGE_KEY) === '1'; } catch { return false; }
  }
  function setEnabled(v) {
    try { localStorage.setItem(STORAGE_KEY, v ? '1' : '0'); } catch {}
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
      .rsfc-grid {
        display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 8px; padding: 8px 0; width: 100%;
      }
      @media (max-width: 900px) { .rsfc-grid { grid-template-columns: repeat(2, 1fr); } }
      .rsfc-card {
        position: relative; background: #0e1014; border-radius: 8px;
        overflow: hidden; border: 1px solid rgba(255,255,255,0.08);
        display: flex; flex-direction: column;
      }
      .rsfc-card .rsfc-head {
        display: flex; align-items: center; justify-content: space-between;
        padding: 4px 8px; font-size: 11px; background: rgba(255,255,255,0.04);
        font-family: ui-monospace, SFMono-Regular, monospace;
      }
      .rsfc-card .rsfc-name { font-weight: 700; text-transform: uppercase; }
      .rsfc-card .rsfc-stat { color: rgba(255,255,255,0.65); }
      .rsfc-card .rsfc-canvas-wrap { position: relative; }
      .rsfc-card canvas { display: block; width: 100%; height: auto; }
      .rsfc-card .rsfc-err {
        position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
        color: #ff7070; background: rgba(0,0,0,0.4); font-size: 12px; padding: 8px; text-align: center;
      }
      .rsfc-card .rsfc-foot {
        padding: 4px 8px; font-size: 11px; color: rgba(255,255,255,0.6);
        border-top: 1px solid rgba(255,255,255,0.06);
      }
      .rsfc-loading {
        padding: 24px; text-align: center; color: rgba(255,255,255,0.6);
        font-size: 13px;
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
    const btn = document.getElementById('tb-settings');
    if (!btn) return null;
    injectStyles();
    menu = document.createElement('div');
    menu.id = 'rsfc-settings-menu';
    menu.className = 'rsfc-menu hidden';
    menu.innerHTML = `
      <label class="rsfc-row">
        <input type="checkbox" id="rsfc-cmp-toggle" ${isEnabled() ? 'checked' : ''}>
        <span class="rsfc-label">Algorithm compare mode</span>
        <span class="rsfc-hint">5 algos in parallel (haar/cnn/yunet/mtcnn/hog)</span>
      </label>
      <div class="rsfc-row rsfc-info">
        <span class="rsfc-hint">When on, image jobs render 5 mini canvases side by side; each canvas is a separate server-side detection run.</span>
      </div>
    `;
    document.body.appendChild(menu);
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
      scheduleCompare();
    });
    return menu;
  }

  let lastJobId = null;
  function scheduleCompare() {
    const enabled = isEnabled();
    const detail = document.getElementById('pv-detail');
    if (!enabled || !detail || detail.classList.contains('hidden')) {
      removeComparePanel();
      return;
    }
    let jobId = null;
    if (window.state && window.state.currentJobId) {
      jobId = window.state.currentJobId;
    } else {
      const idEl = document.getElementById('pv-id');
      if (idEl && idEl.textContent) {
        const m = idEl.textContent.match(/[#]?([0-9a-f-]+)/i);
        if (m) jobId = m[1];
      }
    }
    if (!jobId || jobId === lastJobId) return;
    lastJobId = jobId;
    fetchAndRenderCompare(jobId);
  }

  setInterval(scheduleCompare, 600);

  async function fetchAndRenderCompare(jobId) {
    const host = document.getElementById('pv-stage') || document.getElementById('pv-detail');
    if (!host) return;
    removeComparePanel();
    const panel = document.createElement('div');
    panel.id = 'rsfc-compare-panel';
    panel.className = 'rsfc-loading';
    panel.textContent = 'Running 5 algos in parallel (haar/cnn/yunet/mtcnn/hog)...';
    host.appendChild(panel);
    try {
      const resp = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/compare?algos=${ALGOS.join(',')}`, {
        method: 'POST',
      });
      if (!resp.ok) {
        panel.textContent = `compare failed: HTTP ${resp.status}`;
        return;
      }
      const data = await resp.json();
      const orig = document.getElementById('pv-img');
      const origSrc = orig && orig.src;
      renderComparePanel(panel, data, origSrc);
    } catch (e) {
      panel.textContent = `compare failed: ${e.message || e}`;
    }
  }

  function removeComparePanel() {
    const old = document.getElementById('rsfc-compare-panel');
    if (old) old.remove();
  }

  function renderComparePanel(panel, data, origSrc) {
    panel.classList.remove('rsfc-loading');
    panel.innerHTML = '';
    const head = document.createElement('div');
    head.className = 'rsfc-info';
    head.innerHTML = `<div class="rsfc-row" style="padding:0 0 8px 0;">
      <span class="rsfc-label">Algorithm compare</span>
      <span class="rsfc-hint">${data.width}x${data.height} / ${(data.results || []).length} algos</span>
    </div>`;
    panel.appendChild(head);
    const grid = document.createElement('div');
    grid.className = 'rsfc-grid';
    for (const r of (data.results || [])) {
      grid.appendChild(renderCard(r, origSrc));
    }
    panel.appendChild(grid);
  }

  function renderCard(result, origSrc) {
    const card = document.createElement('div');
    card.className = 'rsfc-card';
    const algo = result.algo;
    const color = ALGO_COLORS[algo] || [255, 255, 255];
    const desc = ALGO_DESC[algo] || algo;
    const head = document.createElement('div');
    head.className = 'rsfc-head';
    head.innerHTML = `
      <span class="rsfc-name" style="color: rgb(${color.join(',')})">${algo}</span>
      <span class="rsfc-stat">${result.detection_count ?? 0} faces / ${result.elapsed_ms ?? 0} ms</span>
    `;
    card.appendChild(head);
    const wrap = document.createElement('div');
    wrap.className = 'rsfc-canvas-wrap';
    const canvas = document.createElement('canvas');
    wrap.appendChild(canvas);
    if (result.error) {
      const err = document.createElement('div');
      err.className = 'rsfc-err';
      err.textContent = result.error;
      wrap.appendChild(err);
    }
    card.appendChild(wrap);
    const foot = document.createElement('div');
    foot.className = 'rsfc-foot';
    foot.textContent = desc;
    card.appendChild(foot);

    if (origSrc) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const w = img.naturalWidth || 320;
        const h = img.naturalHeight || 240;
        canvas.width = w; canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, w, h);
        ctx.lineWidth = Math.max(1.5, w / 200);
        ctx.strokeStyle = `rgb(${color.join(',')})`;
        ctx.font = `${Math.max(10, w / 30)}px ui-monospace, monospace`;
        for (const d of (result.detections || [])) {
          ctx.strokeRect(d.x, d.y, d.w, d.h);
          if (d.score !== undefined) {
            ctx.fillStyle = `rgb(${color.join(',')})`;
            ctx.fillText(d.score.toFixed(2), d.x + 2, d.y + Math.max(12, w / 30));
          }
        }
      };
      img.onerror = () => {
        const ctx = canvas.getContext('2d');
        canvas.width = 320; canvas.height = 240;
        ctx.fillStyle = '#222';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#888';
        ctx.font = '14px sans-serif';
        ctx.fillText('image load failed', 10, 30);
      };
      img.src = origSrc;
    }
    return card;
  }

  function init() {
    ensureSettingsMenu();
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
