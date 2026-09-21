/* rs-face Platform · box-editor.js
 * 人脸框选编辑器:在 lightbox 中允许用户拖动 bbox 修正位置 / 尺寸;
 * 默认客户端保存(写回 state.currentJob.frames[i].faces[j]),
 * 点 💾 按钮 POST 到 /api/frames/:id/correct(若服务端暂未提供则保留本地状态,
 * 等待 server endpoint 落地)。
 *
 * 设计要点:
 *   - 完全独立于 lightbox:另起一层 canvas (#lb-edit-overlay) 叠加在原 overlay 上,
 *     原 drawOverlay() 不动。
 *   - 选用 rail 项(右侧列表)选中要编辑的人脸;canvas 上点击左键也能选中。
 *   - 8 个 handle:四角 + 四边中点(可选拖动缩放);空白处按住中键拖动 = 平移。
 *   - 任何修改只在 _editMap 里;点"恢复"清掉;点"保存"再推回 state。
 *
 * 依赖 utils / lightbox / toast / state(懒引用,定义于 app.js)。
 * 加载顺序:本文件必须在 app.js 之前引入。
 */
'use strict';

const boxEditor = (() => {
  // ---- 状态 -----------------------------------------------------------
  let _editMode = false;          // 当前是否在编辑模式
  let _activeFaceIdx = -1;        // 当前选中的 face index(在 lightbox _frames[_idx].faces 内)
  let _edits = new Map();         // key = `${frameIdx}:${faceIdx}` -> {x,y,w,h, original:{x,y,w,h}}
  let _drag = null;               // {kind:'move'|'nw'|'ne'|'sw'|'se'|'n'|'s'|'e'|'w', startPt, startBox}

  const els = {};                 // DOM 引用

  function ensureOverlay() {
    if (els.editOverlay) return els.editOverlay;
    const stage = utils.$('#lb-stage'); if (!stage) return null;
    const cv = document.createElement('canvas');
    cv.id = 'lb-edit-overlay';
    cv.className = 'lb-edit-overlay hidden';
    cv.setAttribute('aria-hidden', 'true');
    stage.appendChild(cv);
    els.editOverlay = cv;
    return cv;
  }

  /** 从 #lb-meta 解析当前显示的 frame index(轻量,避免改 lightbox 内部 API)。 */
  function currentFrameIndex() {
    const meta = utils.$('#lb-meta');
    if (!meta) return -1;
    const m = /#(\d+)/.exec(meta.textContent || '');
    return m ? parseInt(m[1], 10) : -1;
  }

  function currentFrame() {
    const idx = currentFrameIndex();
    if (idx < 0) return null;
    const job = state.currentJob;
    if (!job) return null;
    return (job.frames || []).find(f => f.index === idx) || null;
  }

  function frameKey(frameIdx, faceIdx) {
    return frameIdx + ':' + faceIdx;
  }

  /** 给定 frame + face 索引,返回当前编辑后(或原值)的 {x,y,w,h}。 */
  function getBox(frameIdx, faceIdx) {
    const e = _edits.get(frameKey(frameIdx, faceIdx));
    if (e) return e;
    const f = currentFrame(); if (!f) return null;
    const face = (f.faces || [])[faceIdx]; if (!face) return null;
    return { x: face.x, y: face.y, w: face.w, h: face.h };
  }

  function setBox(frameIdx, faceIdx, box) {
    _edits.set(frameKey(frameIdx, faceIdx), {
      x: Math.round(box.x), y: Math.round(box.y),
      w: Math.round(box.w), h: Math.round(box.h),
    });
  }

  function clearEditsForFrame(frameIdx) {
    for (const k of Array.from(_edits.keys())) {
      if (k.startsWith(frameIdx + ':')) _edits.delete(k);
    }
  }

  // ---- 坐标转换:屏幕 ↔ 原图像素 ----------------------------------------
  function imgRect() {
    const stage = utils.$('#lb-stage'); const img = utils.$('#lb-img');
    if (!stage || !img) return null;
    const sr = stage.getBoundingClientRect();
    const natW = img.naturalWidth || 1, natH = img.naturalHeight || 1;
    const ar = natW / natH, sar = sr.width / sr.height;
    let dw, dh, dx, dy;
    if (ar > sar) { dw = sr.width; dh = sr.width / ar; dx = 0; dy = (sr.height - dh) / 2; }
    else          { dh = sr.height; dw = sr.height * ar; dy = 0; dx = (sr.width - dw) / 2; }
    return { sr, dw, dh, dx, dy, natW, natH };
  }

  /** 屏幕坐标 -> 原图坐标 */
  function screenToImg(clientX, clientY) {
    const r = imgRect(); if (!r) return null;
    const x = (clientX - r.sr.left - r.dx) / (r.dw / r.natW);
    const y = (clientY - r.sr.top - r.dy) / (r.dh / r.natH);
    return { x, y };
  }

  // ---- 命中测试 -------------------------------------------------------
  const HANDLE_SIZE = 8; // 屏幕像素
  const HANDLES = ['nw', 'ne', 'sw', 'se', 'n', 's', 'e', 'w'];

  function drawHandles(ctx, box, active) {
    // box: {x,y,w,h} in 屏幕像素(已经是 drawOverlay 用的坐标)
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = active ? '#f0c674' : '#94a3b8';
    ctx.fillStyle = active ? '#f0c674' : '#94a3b8';
    const corners = [
      ['nw', box.x,                box.y],
      ['ne', box.x + box.w,        box.y],
      ['sw', box.x,                box.y + box.h],
      ['se', box.x + box.w,        box.y + box.h],
    ];
    for (const [, hx, hy] of corners) {
      ctx.fillRect(hx - HANDLE_SIZE/2, hy - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);
    }
    // 边中点
    const edges = [
      ['n', box.x + box.w/2, box.y],
      ['s', box.x + box.w/2, box.y + box.h],
      ['e', box.x + box.w,   box.y + box.h/2],
      ['w', box.x,           box.y + box.h/2],
    ];
    for (const [, hx, hy] of edges) {
      ctx.fillRect(hx - HANDLE_SIZE/2, hy - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);
    }
    ctx.restore();
  }

  function hitHandle(px, py, box) {
    for (const corner of [
      { name: 'nw', x: box.x,                y: box.y },
      { name: 'ne', x: box.x + box.w,        y: box.y },
      { name: 'sw', x: box.x,                y: box.y + box.h },
      { name: 'se', x: box.x + box.w,        y: box.y + box.h },
    ]) {
      if (Math.abs(px - corner.x) <= HANDLE_SIZE && Math.abs(py - corner.y) <= HANDLE_SIZE) return corner.name;
    }
    for (const edge of [
      { name: 'n', x: box.x + box.w/2, y: box.y },
      { name: 's', x: box.x + box.w/2, y: box.y + box.h },
      { name: 'e', x: box.x + box.w,   y: box.y + box.h/2 },
      { name: 'w', x: box.x,           y: box.y + box.h/2 },
    ]) {
      if (Math.abs(px - edge.x) <= HANDLE_SIZE && Math.abs(py - edge.y) <= HANDLE_SIZE) return edge.name;
    }
    return null;
  }

  // ---- 主绘制 ---------------------------------------------------------
  function redraw() {
    const cv = ensureOverlay(); if (!cv) return;
    const stage = utils.$('#lb-stage'); if (!stage) return;
    const sr = stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    cv.width = sr.width * dpr; cv.height = sr.height * dpr;
    cv.style.width = sr.width + 'px'; cv.style.height = sr.height + 'px';
    const ctx = cv.getContext('2d'); if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, sr.width, sr.height);

    const f = currentFrame(); if (!f) return;
    const r = imgRect(); if (!r) return;
    const sX = r.dw / r.natW, sY = r.dh / r.natH;

    // 画所有 face 的编辑框(虚线);active 用实线 + handles
    const faces = f.faces || [];
    for (let i = 0; i < faces.length; i++) {
      const box = getBox(f.index, i) || faces[i];
      const x = r.dx + (box.x || 0) * sX;
      const y = r.dy + (box.y || 0) * sY;
      const w = (box.w || 0) * sX;
      const h = (box.h || 0) * sY;
      const isActive = i === _activeFaceIdx;
      ctx.save();
      ctx.lineWidth = isActive ? 2 : 1.2;
      ctx.strokeStyle = isActive ? '#f0c674' : 'rgba(240, 198, 116, 0.55)';
      ctx.setLineDash(isActive ? [] : [4, 3]);
      ctx.strokeRect(x, y, w, h);
      ctx.restore();
      if (isActive) drawHandles(ctx, { x, y, w, h }, true);
    }
  }

  // ---- 拖拽处理 -------------------------------------------------------
  function findFaceAt(px, py) {
    const f = currentFrame(); if (!f) return -1;
    const r = imgRect(); if (!r) return -1;
    const sX = r.dw / r.natW, sY = r.dh / r.natH;
    // 从后往前选(后画的优先)
    for (let i = (f.faces || []).length - 1; i >= 0; i--) {
      const box = getBox(f.index, i) || f.faces[i];
      const x = r.dx + (box.x || 0) * sX;
      const y = r.dy + (box.y || 0) * sY;
      const w = (box.w || 0) * sX;
      const h = (box.h || 0) * sY;
      if (px >= x - 4 && px <= x + w + 4 && py >= y - 4 && py <= y + h + 4) return i;
    }
    return -1;
  }

  function onPointerDown(e) {
    if (!_editMode) return;
    const cv = ensureOverlay(); if (!cv || cv.classList.contains('hidden')) return;
    e.preventDefault(); e.stopPropagation();
    const stage = utils.$('#lb-stage');
    const sr = stage.getBoundingClientRect();
    const px = e.clientX - sr.left, py = e.clientY - sr.top;
    const f = currentFrame(); if (!f) return;

    // 先看 active face 的 handle
    if (_activeFaceIdx >= 0) {
      const box = getBox(f.index, _activeFaceIdx);
      const r = imgRect();
      const x = r.dx + (box.x || 0) * (r.dw / r.natW);
      const y = r.dy + (box.y || 0) * (r.dh / r.natH);
      const w = (box.w || 0) * (r.dw / r.natW);
      const h = (box.h || 0) * (r.dh / r.natH);
      const handle = hitHandle(px, py, { x, y, w, h });
      if (handle) {
        _drag = { kind: handle, startPt: screenToImg(e.clientX, e.clientY), startBox: { ...box } };
        stage.setPointerCapture(e.pointerId);
        return;
      }
    }
    // 否则选 face;命中进入 move 拖动
    const hit = findFaceAt(px, py);
    if (hit >= 0) {
      _activeFaceIdx = hit;
      const box = getBox(f.index, hit);
      _drag = { kind: 'move', startPt: screenToImg(e.clientX, e.clientY), startBox: { ...box } };
      stage.setPointerCapture(e.pointerId);
      redraw();
      // 同步右侧 rail 高亮
      syncRailHighlight();
      return;
    }
    // 空白处取消选中
    _activeFaceIdx = -1;
    syncRailHighlight();
    redraw();
  }

  function onPointerMove(e) {
    if (!_editMode || !_drag) return;
    e.preventDefault();
    const f = currentFrame(); if (!f) return;
    const pt = screenToImg(e.clientX, e.clientY);
    if (!pt) return;
    const dx = pt.x - _drag.startPt.x;
    const dy = pt.y - _drag.startPt.y;
    const b = { ..._drag.startBox };
    let nx = b.x, ny = b.y, nw = b.w, nh = b.h;
    const MIN = 8;
    switch (_drag.kind) {
      case 'move':
        nx = b.x + dx; ny = b.y + dy;
        break;
      case 'nw':
        nx = b.x + dx; ny = b.y + dy; nw = b.w - dx; nh = b.h - dy; break;
      case 'ne':
        ny = b.y + dy; nw = b.w + dx; nh = b.h - dy; break;
      case 'sw':
        nx = b.x + dx; nw = b.w - dx; nh = b.h + dy; break;
      case 'se':
        nw = b.w + dx; nh = b.h + dy; break;
      case 'n':
        ny = b.y + dy; nh = b.h - dy; break;
      case 's':
        nh = b.h + dy; break;
      case 'e':
        nw = b.w + dx; break;
      case 'w':
        nx = b.x + dx; nw = b.w - dx; break;
    }
    // 钳制最小尺寸 + 不超图边界
    if (nw < MIN) { if (_drag.kind === 'nw' || _drag.kind === 'sw' || _drag.kind === 'w') nx = b.x + b.w - MIN; nw = MIN; }
    if (nh < MIN) { if (_drag.kind === 'nw' || _drag.kind === 'ne' || _drag.kind === 'n') ny = b.y + b.h - MIN; nh = MIN; }
    nx = Math.max(0, nx); ny = Math.max(0, ny);
    if (nx + nw > (f.original_w || 100000)) nw = Math.max(MIN, (f.original_w || 100000) - nx);
    if (ny + nh > (f.original_h || 100000)) nh = Math.max(MIN, (f.original_h || 100000) - ny);
    redraw();
    // 暂存(还没 save):写入 _edits,等待 onPointerUp 提交
    _pending = { frameIdx: f.index, faceIdx: _activeFaceIdx, box: { x: nx, y: ny, w: nw, h: nh } };
  }

  function onPointerUp(e) {
    if (!_drag) return;
    const stage = utils.$('#lb-stage');
    if (stage) try { stage.releasePointerCapture(e.pointerId); } catch {}
    _drag = null;
    if (_pending) {
      setBox(_pending.frameIdx, _pending.faceIdx, _pending.box);
      _pending = null;
    }
  }
  let _pending = null;

  // ---- rail 联动 -------------------------------------------------------
  function syncRailHighlight() {
    // rail 的 active 态由 lightbox 维护(_activeFace in app.js),我们只能尽量同步:
    // 重渲染 rail 行,active 行加 .lb-edit-pick 类,不与 lightbox 自身的 .active 冲突。
    document.querySelectorAll('.lb-rail-list .lb-rail-item').forEach((it, i) => {
      it.classList.toggle('lb-edit-pick', i === _activeFaceIdx);
    });
  }

  function onRailClick(e) {
    if (!_editMode) return;
    const it = e.target.closest('.lb-rail-item');
    if (!it) return;
    // rail 内文本是 `#i` 形式
    const numEl = it.querySelector('.lb-rail-i');
    const txt = numEl ? numEl.textContent : '';
    const m = /(\d+)/.exec(txt || '');
    if (!m) return;
    const i = parseInt(m[1], 10) - 1;
    _activeFaceIdx = i;
    syncRailHighlight();
    redraw();
  }

  // ---- 启用 / 关闭编辑模式 ---------------------------------------------
  function setEditMode(on) {
    _editMode = !!on;
    const cv = ensureOverlay();
    if (cv) cv.classList.toggle('hidden', !_editMode);
    const toggle = utils.$('#lb-edit-toggle');
    if (toggle) {
      toggle.classList.toggle('primary', _editMode);
      toggle.setAttribute('aria-pressed', _editMode ? 'true' : 'false');
      toggle.textContent = _editMode ? '✎ 编辑中' : '✎ 编辑';
    }
    const save = utils.$('#lb-edit-save'); if (save) save.classList.toggle('hidden', !_editMode);
    const reset = utils.$('#lb-edit-reset'); if (reset) reset.classList.toggle('hidden', !_editMode);
    if (_editMode) {
      // 切到第一张脸
      const f = currentFrame();
      if (f && (f.faces || []).length && _activeFaceIdx < 0) _activeFaceIdx = 0;
      syncRailHighlight();
    }
    redraw();
  }

  function onToggle() { setEditMode(!_editMode); }
  function onReset() {
    const f = currentFrame(); if (!f) return;
    clearEditsForFrame(f.index);
    redraw();
    toast.info('已恢复原框');
  }

  async function onSave() {
    const f = currentFrame(); if (!f) return;
    const edits = [];
    for (let i = 0; i < (f.faces || []).length; i++) {
      const k = frameKey(f.index, i);
      if (_edits.has(k)) edits.push({ face: i, ..._edits.get(k) });
    }
    if (!edits.length) {
      toast.info('没有需要保存的修改');
      return;
    }
    // 写回 state.currentJob.frames[i].faces[j]
    for (const ed of edits) {
      const face = f.faces[ed.face];
      if (face) {
        face.x = ed.x; face.y = ed.y; face.w = ed.w; face.h = ed.h;
        face._corrected = true;
      }
    }
    // 同步侧栏缩略图 overlay
    if (typeof overlay !== 'undefined' && overlay && state.currentJob) overlay.refresh(state.currentJob);
    // 同步 lightbox overlay
    if (typeof lightbox !== 'undefined' && lightbox && lightbox.drawOverlay) {
      try { lightbox.drawOverlay(); } catch {}
    }
    // 尝试 POST 到 /api/frames/:id/correct(服务端未实现时本地已生效)
    // TODO: server endpoint — POST /api/frames/:id/correct { corrections:[{face,x,y,w,h}] }
    try {
      const r = await fetch('/api/frames/' + encodeURIComponent(f.id || f.index) + '/correct', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ corrections: edits.map(f => ({ face: f.face, x: f.x, y: f.y, w: f.w, h: f.h })) }),
      });
      if (r.ok) toast.success(`已保存 ${edits.length} 处修正(服务端)`);
      else toast.warn(`已保存 ${edits.length} 处修正到本地(服务端 ${r.status})`);
    } catch (e) {
      // 网络失败 / 端点未注册 → 静默保留本地;给一个温和提示
      toast.warn(`已保存 ${edits.length} 处修正到本地(服务端暂不可用)`);
    }
  }

  // ---- 键盘快捷键 -----------------------------------------------------
  function onKey(e) {
    if (!_editMode) return;
    const modal = utils.$('#modal-lightbox');
    if (!modal || modal.classList.contains('hidden')) return;
    if (e.target && e.target.matches && e.target.matches('input, textarea')) return;
    const f = currentFrame(); if (!f) return;
    const step = e.shiftKey ? 10 : 1;
    if (_activeFaceIdx < 0) return;
    const box = getBox(f.index, _activeFaceIdx); if (!box) return;
    let changed = false;
    if (e.key === 'ArrowLeft')  { box.x -= step; changed = true; }
    if (e.key === 'ArrowRight') { box.x += step; changed = true; }
    if (e.key === 'ArrowUp')    { box.y -= step; changed = true; }
    if (e.key === 'ArrowDown')  { box.y += step; changed = true; }
    if (e.key === '[')          { box.w = Math.max(8, box.w - step); box.x += step/2; changed = true; }
    if (e.key === ']')          { box.w = box.w + step; box.x -= step/2; changed = true; }
    if (e.key === '-')          { box.h = Math.max(8, box.h - step); box.y += step/2; changed = true; }
    if (e.key === '=')          { box.h = box.h + step; box.y -= step/2; changed = true; }
    if (changed) {
      e.preventDefault();
      setBox(f.index, _activeFaceIdx, box);
      redraw();
    }
  }

  // ---- 初始化 ---------------------------------------------------------
  function init() {
    if (typeof lightbox === 'undefined') return;
    const stage = utils.$('#lb-stage'); if (!stage) return;
    ensureOverlay();
    // 在 stage 上挂一次;pointer events 在 edit-overlay canvas 上接
    stage.addEventListener('pointerdown', onPointerDown);
    stage.addEventListener('pointermove', onPointerMove);
    stage.addEventListener('pointerup', onPointerUp);
    stage.addEventListener('pointercancel', onPointerUp);
    const rail = utils.$('#lb-rail-list');
    if (rail) rail.addEventListener('click', onRailClick);
    const toggle = utils.$('#lb-edit-toggle'); if (toggle) toggle.addEventListener('click', onToggle);
    const reset = utils.$('#lb-edit-reset'); if (reset) reset.addEventListener('click', onReset);
    const save = utils.$('#lb-edit-save'); if (save) save.addEventListener('click', onSave);
    // lightbox 关闭时清掉 active 编辑(避免脏状态)
    const modal = utils.$('#modal-lightbox');
    if (modal) {
      const obs = new MutationObserver(() => {
        if (modal.classList.contains('hidden') && _editMode) {
          setEditMode(false);
          _activeFaceIdx = -1;
          _edits.clear();
        }
      });
      obs.observe(modal, { attributes: true, attributeFilter: ['class'] });
    }
    // 键盘
    document.addEventListener('keydown', onKey);
    // stage 缩放 / 切帧时 重画 handles
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(() => redraw());
      ro.observe(stage);
    }
    // 切换 lightbox 帧 → 重画
    const prev = utils.$('#lb-prev'); const next = utils.$('#lb-next');
    if (prev) prev.addEventListener('click', () => setTimeout(redraw, 50));
    if (next) next.addEventListener('click', () => setTimeout(redraw, 50));
  }

  return { init, setEditMode, redraw, isEditMode: () => _editMode };
})();