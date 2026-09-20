/* rs-face Platform · upload-queue.js
 *
 * 多文件上传队列(零依赖,无构建):
 *  - 把上传从 fire-and-forget 升级为可见队列:文件列表 + 进度条 + 失败重试
 *  - 默认并发 3,可在 #uq-concurrency 调节
 *  - 暴露 uploadQueue.enqueue(files, kind) 给 app.js / dropzone / 粘贴 复用
 *  - 不接管现有 submitImage/submitVideo/submitStream:它们仍是上传函数,
 *    但本模块负责排程、状态展示与失败兜底。
 *
 * DOM 节点(在 index.html 里已就绪):
 *   #upload-queue-panel     容器(浮动面板)
 *   #upload-queue-head      头部(标题 + 折叠按钮)
 *   #upload-queue-list      文件列表(<ul>)
 *   #upload-queue-summary   聚合进度文字 "X / Y · Z MB"
 *   #upload-queue-bar-fill  聚合进度条
 *   #upload-queue-concurrency  并发数显示
 *
 * 加载顺序:本文件必须在 app.js 之前引入(defer 自动按顺序)。
 */
'use strict';

const uploadQueue = (() => {
  const KIND_LABEL = { image: '图片', video: '视频', stream: '流' };
  const STATE = { queued: 'queued', uploading: 'uploading', done: 'done', error: 'error', cancelled: 'cancelled' };
  const DEFAULT_CONCURRENCY = 3;
  const MAX_PARALLEL_HARD_CAP = 8;
  const items = [];   // [{ id, file, kind, status, attempts, error, jobId?, bytesSent, bytesTotal, xhr? }]
  const els = {};     // DOM 缓存
  let _seq = 0;       // item id 自增,DOM 元素 id 也用它
  let _concurrency = DEFAULT_CONCURRENCY;
  let _inflight = 0;  // 当前正在上传的项数
  let _initialized = false;
  let _minimized = false;

  function init() {
    if (_initialized) return;
    _initialized = true;
    els.panel    = utils.$('#upload-queue-panel');
    els.head     = utils.$('#upload-queue-head');
    els.list     = utils.$('#upload-queue-list');
    els.summary  = utils.$('#upload-queue-summary');
    els.barFill  = utils.$('#upload-queue-bar-fill');
    els.concur   = utils.$('#upload-queue-concurrency');
    if (!els.panel || !els.list) return;
    // 读持久化的并发数(用户偏好)
    try {
      const v = parseInt(localStorage.getItem('rsface.upload.concurrency') || '', 10);
      if (v >= 1 && v <= MAX_PARALLEL_HARD_CAP) _concurrency = v;
    } catch {}
    if (els.concur) {
      els.concur.textContent = `并发 ${_concurrency}`;
      els.concur.title = `当前上传并发数 (1-${MAX_PARALLEL_HARD_CAP})`;
      els.concur.addEventListener('click', () => {
        // 1 -> 2 -> 3 -> 4 -> 6 -> 8 -> 1 循环
        const cycle = [1, 2, 3, 4, 6, MAX_PARALLEL_HARD_CAP];
        const i = cycle.indexOf(_concurrency);
        _concurrency = cycle[(i + 1) % cycle.length];
        try { localStorage.setItem('rsface.upload.concurrency', String(_concurrency)); } catch {}
        els.concur.textContent = `并发 ${_concurrency}`;
        pump();
      });
    }
    if (els.head) {
      const btn = els.head.querySelector('.seq-toggle');
      if (btn) btn.addEventListener('click', () => toggleMinimize());
      els.head.addEventListener('click', (e) => {
        if (e.target.closest('.seq-toggle, .seq-clear, .seq-conc, .seq-retry')) return;
        toggleMinimize();
      });
    }
    els.panel.classList.add('seq-hidden');
  }

  function toggleMinimize() {
    _minimized = !_minimized;
    els.panel.classList.toggle('seq-minimized', _minimized);
    const btn = els.panel.querySelector('.seq-toggle');
    if (btn) { btn.textContent = _minimized ? '▢' : '▭'; btn.title = _minimized ? '展开' : '收起'; }
  }

  /**
   * 入队一个或多个文件。
   * @param {FileList|Array<File>} files
   * @param {'image'|'video'|'stream'} kind
   * @returns {Array<string>} 新加入项的 id 列表
   */
  function enqueue(files, kind) {
    init();
    if (!files || !files.length) return [];
    const arr = Array.from(files);
    const newIds = [];
    for (const file of arr) {
      const id = 'uq-' + (++_seq);
      const item = {
        id, file, kind,
        name: file.name || ('unnamed-' + id),
        size: file.size || 0,
        type: file.type || '',
        status: STATE.queued,
        attempts: 0,
        error: null,
        jobId: null,
        bytesSent: 0,
        bytesTotal: file.size || 0,
        xhr: null,
      };
      items.push(item);
      newIds.push(id);
      appendItemRow(item);
    }
    updateSummary();
    showPanel();
    if (window.__track) window.__track('upload_queued', { kind, count: arr.length });
    // 立即尝试调度
    pump();
    return newIds;
  }

  /** 入队一个失败项(用于重试)。 */
  function requeue(id) {
    const it = items.find(x => x.id === id);
    if (!it) return;
    if (it.status !== STATE.error && it.status !== STATE.cancelled) return;
    it.status = STATE.queued;
    it.error = null;
    it.bytesSent = 0;
    updateItemRow(it);
    updateSummary();
    pump();
  }

  /** 取消正在上传或排队的项。 */
  function cancel(id) {
    const it = items.find(x => x.id === id);
    if (!it) return;
    if (it.status === STATE.uploading && it.xhr) {
      try { it.xhr.abort(); } catch {}
      _inflight = Math.max(0, _inflight - 1);
    }
    if (it.status === STATE.done) return;
    it.status = STATE.cancelled;
    it.xhr = null;
    updateItemRow(it);
    updateSummary();
    pump();
  }

  /** 移除已完成 / 已取消 / 已失败项(保留仍在排队的)。 */
  function clearDone() {
    let i = items.length;
    while (i--) {
      const s = items[i].status;
      if (s === STATE.done || s === STATE.cancelled || s === STATE.error) items.splice(i, 1);
    }
    renderList();
    updateSummary();
  }

  function pending() {
    return items.filter(x => x.status === STATE.queued || x.status === STATE.uploading).length;
  }

  function showPanel() {
    if (!els.panel) return;
    els.panel.classList.remove('seq-hidden');
  }

  function hidePanelIfEmpty() {
    if (!items.length && els.panel) els.panel.classList.add('seq-hidden');
  }

  /** 调度器:维持 inflight < concurrency。 */
  function pump() {
    init();
    while (_inflight < _concurrency) {
      const next = items.find(x => x.status === STATE.queued);
      if (!next) break;
      next.status = STATE.uploading;
      next.attempts++;
      _inflight++;
      updateItemRow(next);
      updateSummary();
      runUpload(next).finally(() => {
        _inflight = Math.max(0, _inflight - 1);
        updateSummary();
        pump();
        hidePanelIfEmpty();
      });
    }
  }

  /** 真正发起一次上传。成功 → done;失败 → error;Abort → cancelled。 */
  async function runUpload(it) {
    const url = it.kind === 'image' ? '/api/jobs/image' : (it.kind === 'video' ? '/api/import/video' : null);
    if (!url) {
      // stream kind 不走 multipart,留给原 upload.submitStream;此处直接 done 占位
      try {
        await api.postStream(it.file && it.file.url ? it.file.url : it.name, undefined);
      } catch (e) { fail(it, e.message || 'stream failed'); return; }
      finish(it, null);
      return;
    }
    const fd = new FormData();
    fd.append('file', it.file);
    const algo = currentAlgoChoice();
    if (algo) fd.append('algo', algo);
    try {
      const data = await xhrUpload(url, fd, it);
      if (it.status === STATE.cancelled) return;
      finish(it, data && data.job_id ? data.job_id : null);
    } catch (e) {
      if (it.status === STATE.cancelled) return;
      fail(it, e && e.message ? e.message : 'upload failed');
    }
  }

  /** 用 XHR 而非 fetch:能拿到真实的 bytesSent / 进度事件。 */
  function xhrUpload(url, fd, it) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      it.xhr = xhr;
      xhr.open('POST', url, true);
      xhr.upload.onprogress = (e) => {
        if (!e.lengthComputable) return;
        it.bytesSent = e.loaded;
        updateItemRow(it);
      };
      xhr.onload = () => {
        let body = null;
        try { body = xhr.responseText ? JSON.parse(xhr.responseText) : null; } catch {}
        if (xhr.status >= 200 && xhr.status < 300) {
          if (body && body.error) { reject(new Error(body.error)); return; }
          resolve(body || {});
        } else {
          reject(new Error((body && body.error) || ('HTTP ' + xhr.status)));
        }
      };
      xhr.onerror = () => reject(new Error('network error'));
      xhr.onabort = () => {
        const e = new Error('aborted'); e.cancelled = true; reject(e);
      };
      xhr.send(fd);
    });
  }

  function finish(it, jobId) {
    it.status = STATE.done;
    it.jobId = jobId || null;
    it.xhr = null;
    updateItemRow(it);
    if (jobId) {
      // 复用 upload 模块的副作用:刷新 sidebar / preview
      try {
        api.getJob(jobId).then(job => {
          if (!job) return;
          sidebar.upsertJob(job);
          // 第一个完成的项自动打开预览,其它静默
          if (items.filter(x => x.status === STATE.done).length === 1) {
            preview.open(job.id);
          }
        }).catch(() => {});
      } catch {}
    }
    if (window.__track) window.__track('upload_done', { kind: it.kind, size_kb: Math.round((it.size || 0) / 1024), job_id: !!jobId });
  }

  function fail(it, msg) {
    it.status = STATE.error;
    it.error = msg;
    it.xhr = null;
    updateItemRow(it);
    toast.error(`上传失败: ${it.name} · ${msg}`);
    if (window.__track) window.__track('upload_failed', { kind: it.kind, message: (msg || '').slice(0, 64) });
  }

  function currentAlgoChoice() {
    const sel = utils.$('#new-algo-select');
    if (!sel) return undefined;
    const v = (sel.value || '').trim();
    return v ? v : undefined;
  }

  /** ------- DOM ------- */
  function appendItemRow(it) {
    if (!els.list) return;
    const li = document.createElement('li');
    li.className = 'seq-item';
    li.id = it.id;
    li.dataset.status = it.status;
    li.innerHTML = `
      <div class="seq-row1">
        <span class="seq-kind">${escape(KIND_LABEL[it.kind] || it.kind)}</span>
        <span class="seq-name" title="${escape(it.name)}">${escape(it.name)}</span>
        <span class="seq-size">${humanSize(it.size)}</span>
        <span class="seq-status">${statusText(it)}</span>
        <span class="seq-act"></span>
      </div>
      <div class="seq-row2"><div class="seq-bar"><div class="seq-bar-fill"></div></div></div>
    `;
    els.list.appendChild(li);
    wireRowActions(li, it);
    updateItemRow(it);
  }

  function updateItemRow(it) {
    const li = els.list && els.list.querySelector('#' + it.id);
    if (!li) return;
    li.dataset.status = it.status;
    const statusEl = li.querySelector('.seq-status');
    if (statusEl) statusEl.textContent = statusText(it);
    const barFill = li.querySelector('.seq-bar-fill');
    if (barFill) {
      let pct = 0;
      if (it.status === STATE.uploading && it.bytesTotal > 0) {
        pct = Math.min(99, Math.round((it.bytesSent / it.bytesTotal) * 100));
      } else if (it.status === STATE.done) {
        pct = 100;
      } else if (it.status === STATE.error || it.status === STATE.cancelled) {
        pct = 0;
      }
      barFill.style.width = pct + '%';
      barFill.dataset.state = it.status;
    }
    const actEl = li.querySelector('.seq-act');
    if (actEl) {
      actEl.innerHTML = '';
      const mk = (cls, txt, title, fn) => {
        const b = document.createElement('button');
        b.className = cls; b.type = 'button';
        b.textContent = txt; b.title = title || txt;
        b.addEventListener('click', (e) => { e.stopPropagation(); fn(); });
        return b;
      };
      if (it.status === STATE.queued || it.status === STATE.uploading) {
        actEl.appendChild(mk('seq-btn seq-x', '×', '取消', () => cancel(it.id)));
      } else if (it.status === STATE.error) {
        actEl.appendChild(mk('seq-btn seq-retry', '↻', '重试', () => requeue(it.id)));
        actEl.appendChild(mk('seq-btn seq-x', '×', '移除', () => remove(it.id)));
      } else if (it.status === STATE.done) {
        actEl.appendChild(mk('seq-btn seq-x', '×', '移除', () => remove(it.id)));
      }
    }
    // 错误时整行 tooltip
    if (it.error) {
      li.title = it.error;
    } else {
      li.removeAttribute('title');
    }
  }

  function remove(id) {
    const i = items.findIndex(x => x.id === id);
    if (i < 0) return;
    if (items[i].status === STATE.uploading) cancel(id);
    items.splice(i, 1);
    const li = els.list && els.list.querySelector('#' + id);
    if (li) li.remove();
    updateSummary();
    hidePanelIfEmpty();
  }

  function renderList() {
    if (!els.list) return;
    els.list.innerHTML = '';
    for (const it of items) appendItemRow(it);
  }

  function wireRowActions(li, it) {
    li.addEventListener('click', (e) => {
      if (e.target.closest('.seq-btn')) return;
      if (it.jobId) preview.open(it.jobId);
    });
  }

  function updateSummary() {
    if (!els.summary || !els.barFill) return;
    const total = items.length;
    const done = items.filter(x => x.status === STATE.done).length;
    const err = items.filter(x => x.status === STATE.error).length;
    const cancel = items.filter(x => x.status === STATE.cancelled).length;
    const totalSize = items.reduce((a, b) => a + (b.size || 0), 0);
    const sentSize = items.reduce((a, b) => a + (b.status === STATE.done ? b.size : (b.status === STATE.uploading ? b.bytesSent : 0)), 0);
    els.summary.textContent = total === 0
      ? '空'
      : `${done} / ${total} 完成 · ${humanSize(sentSize)} / ${humanSize(totalSize)} · 失败 ${err} · 取消 ${cancel}`;
    const pct = total === 0 ? 0 : Math.round((done / total) * 100);
    els.barFill.style.width = pct + '%';
    const remain = items.filter(x => x.status === STATE.queued || x.status === STATE.uploading).length;
    if (els.concur) {
      els.concur.textContent = remain > 0 ? `并发 ${_concurrency} · 进行 ${remain}` : `并发 ${_concurrency}`;
    }
    if (total === 0) {
      els.panel.classList.add('seq-hidden');
    }
  }

  function statusText(it) {
    if (it.status === STATE.queued) return '等待';
    if (it.status === STATE.uploading) {
      const pct = it.bytesTotal > 0 ? Math.round((it.bytesSent / it.bytesTotal) * 100) : 0;
      return `${pct}%`;
    }
    if (it.status === STATE.done) return '完成';
    if (it.status === STATE.error) return '失败';
    if (it.status === STATE.cancelled) return '已取消';
    return it.status;
  }

  function humanSize(n) {
    if (n == null || isNaN(n)) return '0 B';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + ' MB';
    return (n / 1024 / 1024 / 1024).toFixed(2) + ' GB';
  }

  function escape(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  /** Test / debug helper:当前队列快照。 */
  function snapshot() {
    return items.map(x => ({ id: x.id, name: x.name, status: x.status, jobId: x.jobId }));
  }

  return { init, enqueue, requeue, cancel, remove, clearDone, pending, snapshot };
})();

// DOMContentLoaded 后初始化;index.html 里 defer 保证 DOM 已就绪。
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => uploadQueue.init(), { once: true });
} else {
  uploadQueue.init();
}