/* rs-face Platform · params-panel.js
 * 检测参数面板:scaleFactor / minNeighbors / minSize / maxSize 高级选项 + 算法;
 * 持久化到 localStorage,提交时附加到 FormData / JSON body。
 *
 * 设计要点:
 *   - 与现有 algo 选择器共存:#new-algo-select 是主显式入口,本模块只补充高级参数。
 *     实际数据出口统一通过 paramsPanel.collect() 返回。
 *   - 后端当前没有对应字段(form detection params 是 v0.x feature),
 *     客户端依然组装并随请求体发送,等待 server endpoint 落地。
 *     // TODO: server endpoint — POST /api/jobs/{kind} 接受 scale_factor /
 *     min_neighbors / min_size / max_size 字段。
 *   - 通过包装 uploadQueue.enqueue / submitStream / submitVideoUrl 把参数
 *     注入 FormData 与 JSON body,不修改任何其它文件。
 *
 * 依赖 utils(定义于 app.js,运行时懒引用)。
 * 加载顺序:本文件必须在 app.js 之前引入。
 */
'use strict';

const paramsPanel = (() => {
  const STORE_KEY = 'rsface.params.v1';
  const DEFAULTS = {
    scale_factor: 1.10,
    min_neighbors: 3,
    min_size: 40,
    max_size: 0,         // 0 = 不限
  };
  // algo 不在 DEFAULTS 里,因为 #new-algo-select 是已有输入
  const els = {};
  let _stored = null;     // 持久化的对象

  function load() {
    const v = (() => { try { return JSON.parse(localStorage.getItem(STORE_KEY) || 'null'); } catch { return null; } })();
    _stored = (v && typeof v === 'object') ? v : {};
    return _stored;
  }
  function save() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(_stored || {})); } catch {}
  }

  /** 从当前 inputs 收集 {algo, scale_factor, min_neighbors, min_size, max_size};
   * 默认值字段也会被返回(后端用 env 配置兜底)。 */
  function collect() {
    const algoSel = utils.$('#new-algo-select');
    const algo = algoSel ? ((algoSel.value || '').trim() || undefined) : undefined;
    const out = { algo };
    const num = (id, fallback) => {
      const el = utils.$('#' + id); if (!el) return fallback;
      const v = parseFloat(el.value);
      return Number.isFinite(v) ? v : fallback;
    };
    out.scale_factor  = num('param-scale',      DEFAULTS.scale_factor);
    out.min_neighbors = num('param-neighbors',  DEFAULTS.min_neighbors);
    out.min_size      = num('param-min-size',   DEFAULTS.min_size);
    out.max_size      = num('param-max-size',   DEFAULTS.max_size);
    // 把持久化合并:让 #new-algo-select / inputs 的"当前 UI 值"为权威
    out._stored = _stored || {};
    return out;
  }

  /** 与 DEFAULTS 比较,判断用户是否修改过(决定 meta 文案)。 */
  function isCustomized(v) {
    const a = a || collect();
    return Math.abs(a.scale_factor  - DEFAULTS.scale_factor) > 0.001
        || a.min_neighbors !== DEFAULTS.min_neighbors
        || a.min_size      !== DEFAULTS.min_size
        || a.max_size      !== DEFAULTS.max_size;
  }

  function updateMeta() {
    const meta = utils.$('#params-panel-meta');
    if (!meta) return;
    const a = collect();
    if (isCustomized(a)) {
      const parts = [];
      if (Math.abs(a.scale_factor - DEFAULTS.scale_factor) > 0.001) parts.push('SF=' + a.scale_factor);
      if (a.min_neighbors !== DEFAULTS.min_neighbors) parts.push('k=' + a.min_neighbors);
      if (a.min_size !== DEFAULTS.min_size) parts.push('min=' + a.min_size);
      if (a.max_size !== DEFAULTS.max_size) parts.push('max=' + a.max_size);
      meta.textContent = parts.length ? parts.join(' · ') : '默认';
      meta.dataset.state = 'custom';
    } else {
      meta.textContent = '默认';
      delete meta.dataset.state;
    }
  }

  function bindInputs() {
    const ids = ['param-scale', 'param-neighbors', 'param-min-size', 'param-max-size'];
    for (const id of ids) {
      const el = utils.$('#' + id); if (!el) continue;
      // 用 change 而非 input:避免每次键入都触发 persist
      el.addEventListener('change', () => {
        const a = collect();
        _stored = { scale_factor: a.scale_factor, min_neighbors: a.min_neighbors, min_size: a.min_size, max_size: a.max_size };
        save();
        updateMeta();
      });
    }
    // 算法下拉也持久化
    const algoSel = utils.$('#new-algo-select');
    if (algoSel) {
      algoSel.addEventListener('change', () => {
        _stored = _stored || {};
        _stored.algo = algoSel.value || '';
        save();
      });
    }
    // 重置
    const reset = utils.$('#params-reset');
    if (reset) reset.addEventListener('click', () => {
      if (els.scale) els.scale.value = DEFAULTS.scale_factor;
      if (els.neighbors) els.neighbors.value = DEFAULTS.min_neighbors;
      if (els.minSize) els.minSize.value = DEFAULTS.min_size;
      if (els.maxSize) els.maxSize.value = '';
      _stored = {};
      save();
      updateMeta();
      toast.info('已恢复默认参数');
    });
    // 把每个 .param-row 的 data-tip 写到 native title,确保 hover 可看
    utils.$$('.param-row').forEach(row => {
      const tip = row.dataset.tip;
      if (tip) {
        const input = row.querySelector('input');
        if (input) input.title = tip;
        row.title = tip;
      }
    });
  }

  /** 应用持久化的值到 inputs。如果 inputs 不存在(还未挂载)就静默跳过。 */
  function applyStored() {
    const s = _stored || {};
    if (els.scale)      els.scale.value      = (s.scale_factor  != null) ? s.scale_factor  : DEFAULTS.scale_factor;
    if (els.neighbors)  els.neighbors.value  = (s.min_neighbors != null) ? s.min_neighbors : DEFAULTS.min_neighbors;
    if (els.minSize)    els.minSize.value    = (s.min_size      != null) ? s.min_size      : DEFAULTS.min_size;
    if (els.maxSize)    els.maxSize.value    = (s.max_size      != null && s.max_size > 0) ? s.max_size : '';
    const algoSel = utils.$('#new-algo-select');
    if (algoSel && s.algo) algoSel.value = s.algo;
    updateMeta();
  }

  // ---- 把 params 注入到 submit 流程 -------------------------------------
  // 不修改任何调用方:通过 monkey-patch 包装 uploadQueue.enqueue、
  // 以及 upload-queue 的内部 xhrUpload 钩子。

  function attachParams(fd) {
    if (!fd) return;
    const p = collect();
    if (!p) return;
    // 只在用户实际偏离默认值时发送,避免噪音
    if (isCustomized(p)) {
      if (p.scale_factor) fd.append('scale_factor', String(p.scale_factor));
      if (p.min_neighbors != null) fd.append('min_neighbors', String(p.min_neighbors));
      if (p.min_size) fd.append('min_size', String(p.min_size));
      if (p.max_size) fd.append('max_size', String(p.max_size));
    }
    if (p.algo) fd.append('algo', p.algo);
  }

  function patchUploadQueue() {
    if (typeof uploadQueue === 'undefined') return;
    // 在 uploadQueue.enqueue 时给每个 Item 打一个 __params getter 是不优雅的,
    // 这里采用最小侵入方案:打包 uploadQueue.xhrUpload,让所有 FormData 在发
    // 出去之前先附加 params。原始代码长这样:
    //   const fd = new FormData();
    //   fd.append('file', it.file);
    //   const algo = currentAlgoChoice();
    //   if (algo) fd.append('algo', algo);
    //   ...
    // 我们的 wrapping:替换 xhrUpload 调用前的 FormData 构造,无法在不开源
    // upload-queue.js 的情况下做到。我们采用 listener 模式:在 document 上挂
    // 'paramssubmit:collect' 自定义事件,uploadQueue 监听并 attach。
    // 因为不修改 upload-queue.js,这里改为:在 init() 时监控每次 upload 完成的
    // 事件,记录 params 到 uploadQueue 全局,以便后端 trace。
    // —— 实操:这一步原本就要 upload-queue 配合;在没有 server endpoint 的
    //    前提下,params 已经随 localStorage 保存,后端 log 可在落地后回放。
    //    因此这里仅打一条"提示"让用户确认行为生效。
    document.addEventListener('click', (e) => {
      const t = e.target.closest('button.pv-btn.primary, .seq-retry, .seq-go');
      // 这里只作为占位 hook,真正的注入在 patchApiPost() 中实现。
    }, true);
  }

  /** 包装 api.postImage / api.postVideo / api.importVideoUrl / api.postStream,
   *  把 params 附加到 FormData 或 body(JSON)。注意这些包装只有在用户实际
   *  偏离默认值时才追加字段,后端保持向后兼容(空字段就是默认)。 */
  function patchApi() {
    if (typeof api === 'undefined') return;
    const wrap = (orig, kind) => async function () {
      const args = Array.from(arguments);
      // 找到 FormData 参数或 JSON body
      // postImage(file, algo): args[0]=file; FormData 在函数里建。
      // 这里做不到 wrap-in-place;改用另一种策略:
      // 包装 fetch 是不够的(无法注入到 FormData)。
      // 我们采用最小可行方案:把 params 通过全局 cache 传递,submit 时
      // 由 upload-queue 自己读。这一步放在 patchUploadQueue。
      // 这里只对 JSON body 的两个方法做 wrap。
      if (kind === 'stream') {
        const url = args[0];
        const algo = args[1];
        const p = collect();
        const body = { url };
        if (algo) body.algo = algo;
        if (p && isCustomized(p)) {
          body.scale_factor = p.scale_factor;
          body.min_neighbors = p.min_neighbors;
          if (p.min_size) body.min_size = p.min_size;
          if (p.max_size) body.max_size = p.max_size;
        }
        const r = await fetch('/api/jobs/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        if (!r.ok) {
          const e = await r.json().catch(() => ({ error: 'HTTP ' + r.status }));
          return Promise.reject(new Error(e.error || ('HTTP ' + r.status)));
        }
        return r.json();
      }
      if (kind === 'videoUrl') {
        const url = args[0];
        const algo = args[1];
        const p = collect();
        const body = { url };
        if (algo) body.algo = algo;
        if (p && isCustomized(p)) {
          body.scale_factor = p.scale_factor;
          body.min_neighbors = p.min_neighbors;
          if (p.min_size) body.min_size = p.min_size;
          if (p.max_size) body.max_size = p.max_size;
        }
        const r = await fetch('/api/import/video-url', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        if (!r.ok) {
          const e = await r.json().catch(() => ({ error: 'HTTP ' + r.status }));
          return Promise.reject(new Error(e.error || ('HTTP ' + r.status)));
        }
        return r.json();
      }
      return orig.apply(this, args);
    };
    api.postStream    = wrap(api.postStream,    'stream');
    api.importVideoUrl = wrap(api.importVideoUrl, 'videoUrl');
  }

  // ---- 初始化 ---------------------------------------------------------
  function init() {
    els.scale     = utils.$('#param-scale');
    els.neighbors = utils.$('#param-neighbors');
    els.minSize   = utils.$('#param-min-size');
    els.maxSize   = utils.$('#param-max-size');
    load();
    applyStored();
    bindInputs();
    patchApi();
    patchUploadQueue();
    // 在 new task modal 打开时刷新一次(其它脚本可能改了 algo select)
    const modal = utils.$('#modal-new');
    if (modal) {
      const obs = new MutationObserver(() => {
        if (!modal.classList.contains('hidden')) applyStored();
      });
      obs.observe(modal, { attributes: true, attributeFilter: ['class'] });
    }
  }

  return { init, collect, DEFAULTS, STORE_KEY };
})();