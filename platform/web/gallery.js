/* rs-face Platform · 人脸库(persons)管理
 * ----------------------------------------------------------------------------
 * 零依赖 vanilla JS,对接后端已就绪的 /api/persons* CRUD(见
 * server/src/gallery_handlers.rs)。以一个模态承载,刻意不破坏主页的
 * 「单视图」极简结构:顶栏入口按钮 → 打开本模态。
 *
 * 能力:
 *   - 列出全部未归档人员 + 每人已录入人脸数
 *   - 新建人员(display_name 必填;external_id / note 可选)
 *   - 给人员录入人脸(上传图片,multipart `file`)
 *   - 归档(软删除)人员 / 删除单张人脸
 *   - 上传一张图做 1:N 识别测试,展示 top-k 命中与相似度
 *
 * 鉴权:所有请求走被 auth.js 包装过的全局 fetch / FormData,启用了
 * RSFACE_API_TOKEN 时会自动带上 Bearer,无需这里处理。
 */
'use strict';

const galleryUI = (() => {
  let persons = [];
  let nameById = new Map();
  let expanded = new Set(); // 当前展开(加载了 faces-list)的 person id
  let busy = false;

  const el = () => utils.$('#modal-gallery');
  const listEl = () => utils.$('#gal-list');
  const identEl = () => utils.$('#gal-identify-result');

  // ---------- 后端访问 ----------

  async function apiJson(url, opts) {
    const r = await fetch(url, opts);
    if (r.status === 204) return null;
    const text = await r.text();
    let body = null;
    try { body = text ? JSON.parse(text) : null; } catch { body = { error: text }; }
    if (!r.ok) {
      const msg = utils.explainError(body, 'HTTP ' + r.status);
      const code = body && body.error_code ? ` (${body.error_code})` : '';
      throw new Error(msg + code);
    }
    return body;
  }

  async function loadPersons() {
    const data = await apiJson('/api/persons'); // 默认不含已归档
    persons = (data && data.persons) || [];
    nameById = new Map(persons.map(p => [p.id, p.display_name]));
  }

  async function loadFaces(personId) {
    const data = await apiJson(`/api/persons/${encodeURIComponent(personId)}/faces-list`);
    return (data && data.faces) || [];
  }

  // ---------- 渲染 ----------

  function personCard(p) {
    const isOpen = expanded.has(p.id);
    const facesSlot = isOpen ? `<div class="gal-faces" data-faces="${p.id}"><div class="gal-faces-loading">加载中…</div></div>` : '';
    return `
      <div class="gal-person" data-person="${p.id}">
        <div class="gal-person-row">
          <button class="gal-caret" data-act="toggle" data-id="${p.id}" aria-expanded="${isOpen}">${isOpen ? '▾' : '▸'}</button>
          <div class="gal-person-main">
            <span class="gal-person-name">${utils.escapeHtml(p.display_name)}</span>
            ${p.external_id ? `<span class="gal-person-ext mono">${utils.escapeHtml(p.external_id)}</span>` : ''}
          </div>
          <span class="gal-face-count">${p.faces_count} 张</span>
          <div class="gal-person-ops">
            <button class="gal-mini" data-act="enroll" data-id="${p.id}" title="录入人脸">＋脸</button>
            <button class="gal-mini danger" data-act="archive" data-id="${p.id}" title="归档人员">归档</button>
          </div>
        </div>
        ${p.note ? `<div class="gal-person-note">${utils.escapeHtml(p.note)}</div>` : ''}
        ${facesSlot}
      </div>`;
  }

  function render() {
    const host = listEl();
    if (!host) return;
    if (!persons.length) {
      host.innerHTML = `<div class="gal-empty">人脸库为空。先在下方「新建人员」,然后给 TA 录入一张或多张人脸。</div>`;
      return;
    }
    host.innerHTML = persons.map(personCard).join('');
    // 展开的卡片需要拉取 faces
    for (const id of expanded) {
      refreshFaces(id);
    }
  }

  async function refreshFaces(personId) {
    const slot = listEl().querySelector(`[data-faces="${cssEsc(personId)}"]`);
    if (!slot) return;
    try {
      const faces = await loadFaces(personId);
      if (!faces.length) {
        slot.innerHTML = `<div class="gal-faces-empty">还没有录入人脸</div>`;
        return;
      }
      slot.innerHTML = `
        <div class="gal-face-grid">
          ${faces.map(f => `
            <div class="gal-face" data-face="${f.id}">
              <div class="gal-face-thumb">${faceThumb(f)}</div>
              <div class="gal-face-meta">
                <span title="余弦/质量">q ${f.quality.toFixed(2)}</span>
                <span>${f.bbox_w}×${f.bbox_h}</span>
              </div>
              <button class="gal-face-del" data-act="del-face" data-face="${f.id}" data-person="${personId}" title="删除该人脸">✕</button>
            </div>`).join('')}
        </div>`;
    } catch (e) {
      slot.innerHTML = `<div class="gal-faces-empty error">${utils.escapeHtml(e.message)}</div>`;
    }
  }

  /** 有原图就显示缩略图,否则用占位 glyph。 */
  function faceThumb(f) {
    if (f.source_key) {
      return `<img src="${utils.mediaUrl(f.source_key)}" alt="face" loading="lazy">`;
    }
    return `<span class="gal-face-glyph">◗</span>`;
  }

  // ---------- 动作 ----------

  async function createPerson() {
    const name = utils.$('#gal-new-name').value.trim();
    if (!name) { utils.toast('请填写姓名', true); return; }
    const body = {
      display_name: name,
      external_id: utils.$('#gal-new-ext').value.trim() || null,
      note: utils.$('#gal-new-note').value.trim() || null,
    };
    setBusy(true);
    try {
      const p = await apiJson('/api/persons', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      utils.$('#gal-new-name').value = '';
      utils.$('#gal-new-ext').value = '';
      utils.$('#gal-new-note').value = '';
      utils.toast(`已创建「${p.display_name}」`);
      await reload();
    } catch (e) {
      utils.toast('创建失败: ' + e.message, true);
    } finally {
      setBusy(false);
    }
  }

  async function enrollFace(personId, file) {
    const fd = new FormData();
    fd.append('file', file);
    setBusy(true);
    try {
      await apiJson(`/api/persons/${encodeURIComponent(personId)}/faces`, { method: 'POST', body: fd });
      utils.toast('人脸已录入');
      expanded.add(personId);
      await reload(); // reload 内部会 render,展开态保留并触发 faces 拉取
    } catch (e) {
      utils.toast('录入失败: ' + friendlyEnroll(e.message), true);
    } finally {
      setBusy(false);
    }
  }

  async function archivePerson(personId) {
    const name = nameById.get(personId) || personId;
    if (!window.confirm(`归档「${name}」?该人员不再参与识别,数据保留可恢复(后端为软删除)。`)) return;
    setBusy(true);
    try {
      await apiJson(`/api/persons/${encodeURIComponent(personId)}`, { method: 'DELETE' });
      expanded.delete(personId);
      utils.toast(`已归档「${name}」`);
      await reload();
    } catch (e) {
      utils.toast('归档失败: ' + e.message, true);
    } finally {
      setBusy(false);
    }
  }

  async function deleteFace(faceId, personId) {
    if (!window.confirm('删除这张人脸?')) return;
    setBusy(true);
    try {
      await apiJson(`/api/faces/${encodeURIComponent(faceId)}`, { method: 'DELETE' });
      utils.toast('已删除');
      await reload(); // reload 内部会 render,展开态保留并刷新 faces
    } catch (e) {
      utils.toast('删除失败: ' + e.message, true);
    } finally {
      setBusy(false);
    }
  }

  async function identify(file) {
    const out = identEl();
    out.classList.remove('hidden');
    out.innerHTML = '<div class="gal-ident-status">识别中…</div>';
    const fd = new FormData();
    fd.append('file', file);
    fd.append('top_k', '5');
    try {
      const res = await apiJson('/api/identify', { method: 'POST', body: fd });
      renderIdentify(res, file);
    } catch (e) {
      out.innerHTML = `<div class="gal-ident-status error">识别失败:${utils.escapeHtml(friendlyEnroll(e.message))}</div>`;
    }
  }

  function renderIdentify(res, file) {
    const out = identEl();
    const url = URL.createObjectURL(file);
    if (!res.face_count) {
      out.innerHTML = `<div class="gal-ident-status">图中未检测到人脸。</div>`;
      return;
    }
    const faces = res.faces || [];
    const cards = faces.map((f, i) => {
      const blocked = f.blocked;
      const rows = (f.matches || []).map(([pid, score]) => {
        const nm = nameById.get(pid) || (persons.some(p => p.id === pid) ? '(未命名)' : pid);
        const hit = score >= res.threshold;
        return `<div class="gal-match ${hit ? 'hit' : 'miss'}">
          <span class="gal-match-name">${utils.escapeHtml(nm)}</span>
          <span class="gal-match-score mono">${score.toFixed(3)}</span>
        </div>`;
      }).join('') || `<div class="gal-match miss"><span>无候选</span></div>`;
      const live = f.liveness ? `· 活体:${f.liveness.is_real ? '真' : '假'}` : '';
      return `
        <div class="gal-ident-face">
          <div class="gal-ident-index">#${i + 1} ${f.w}×${f.h} ${live}${blocked ? ' · <b>被活体拦截</b>' : ''}</div>
          ${blocked ? '' : rows}
        </div>`;
    }).join('');
    out.innerHTML = `
      <div class="gal-ident-head">
        <img src="${url}" alt="probe" class="gal-ident-img">
        <div class="gal-ident-meta">
          <div>检测到 ${res.face_count} 张脸 · 库内 ${res.gallery_identities} 人 / ${res.gallery_embeddings} 张脸</div>
          <div class="gal-threshold">通过阈值 ${res.threshold.toFixed(2)}（绿色=通过）</div>
        </div>
      </div>
      <div class="gal-ident-list">${cards}</div>`;
  }

  function friendlyEnroll(msg) {
    if (/no_face/i.test(msg)) return '未在图中检测到人脸,请换一张清晰正脸';
    if (/decode_failed/i.test(msg)) return '图片解码失败(仅支持 PNG/JPG/PGM/PPM)';
    if (/file_empty/i.test(msg)) return '文件为空';
    if (/file_too_large/i.test(msg)) return '文件超过图片上传上限';
    return msg;
  }

  // ---------- 骨架 ----------

  function setBusy(b) {
    busy = b;
    const card = el() && el().querySelector('.modal-card');
    if (card) card.classList.toggle('gal-busy', b);
  }

  async function reload() {
    await loadPersons();
    // 清理已不存在的展开态
    const ids = new Set(persons.map(p => p.id));
    for (const id of [...expanded]) if (!ids.has(id)) expanded.delete(id);
    render();
  }

  async function open() {
    modalKit.open('#modal-gallery');
    identEl().classList.add('hidden');
    identEl().innerHTML = '';
    try {
      await reload();
    } catch (e) {
      listEl().innerHTML = `<div class="gal-empty error">加载失败:${utils.escapeHtml(e.message)}<br>（人脸库可能尚未初始化 / 后端未启动）</div>`;
    }
  }

  function onListClick(e) {
    if (busy) return;
    const btn = e.target.closest('[data-act]');
    if (!btn) return;
    const act = btn.dataset.act;
    const id = btn.dataset.id;
    if (act === 'toggle') {
      if (expanded.has(id)) expanded.delete(id);
      else { expanded.add(id); }
      render();
      if (expanded.has(id)) refreshFaces(id);
    } else if (act === 'archive') {
      archivePerson(id);
    } else if (act === 'enroll') {
      pickImage(file => enrollFace(id, file));
    } else if (act === 'del-face') {
      deleteFace(btn.dataset.face, btn.dataset.person);
    }
  }

  let fileInput = null;
  function pickImage(cb) {
    if (!fileInput) {
      fileInput = document.createElement('input');
      fileInput.type = 'file';
      fileInput.accept = 'image/*';
      fileInput.style.display = 'none';
      document.body.appendChild(fileInput);
    }
    fileInput.value = '';
    fileInput.onchange = () => { const f = fileInput.files[0]; if (f) cb(f); };
    fileInput.click();
  }

  function cssEsc(s) {
    return String(s).replace(/[^a-zA-Z0-9_-]/g, c => '\\' + c);
  }

  function init() {
    const top = utils.$('#tb-theme');
    if (top) {
      const b = document.createElement('button');
      b.id = 'tb-gallery';
      b.className = 'tb-icon-btn';
      b.title = '人脸库';
      b.setAttribute('aria-label', '人脸库');
      b.textContent = '👤';
      top.insertAdjacentElement('beforebegin', b);
      b.addEventListener('click', open);
    }
    utils.$('#gal-create')?.addEventListener('click', createPerson);
    utils.$('#gal-identify-pick')?.addEventListener('click', () => pickImage(identify));
    listEl()?.addEventListener('click', onListClick);
  }

  return { init, open };
})();

document.addEventListener('DOMContentLoaded', galleryUI.init);
