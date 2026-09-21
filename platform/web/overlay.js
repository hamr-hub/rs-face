/* rs-face Platform · overlay.js
 * 实时检测可视化:在侧栏任务卡片缩略图上叠加 bounding boxes;
 * 颜色按算法区分(HAAR / LUMINANCE / CNN / ENSEMBLE)。
 *
 * 设计要点:
 *   - 完全独立于 sidebar 渲染:不破坏 updateItem 的签名缓存逻辑
 *   - 仅当 thumbnail 是 currentJob 或 state.currentJobId 匹配时才叠加,避免
 *     200 个 job 全部重绘的开销
 *   - SSE 帧事件 / 心跳事件 触发 refresh():如果 currentJob 有 bbox 数据,
 *     就在它的 sb-thumb 里 draw
 *   - 点击 thumb 已经走 sidebar 的 click → preview.open,这里不重复挂事件
 *
 * 依赖 utils / sidebar / state(懒引用,定义于 app.js / sidebar.js)。
 * 加载顺序:本文件必须在 app.js 之前引入。
 */
'use strict';

const overlay = (() => {
  // 算法 → 描边颜色。沿用 lightbox 已有的调色板,保证视觉一致。
  const COLORS = {
    haar:      '#4fc3f7', // sky-400
    luminance: '#34d399', // green-400
    cnn:       '#f87171', // red-400
    ensemble:  '#f0c674', // gold (highlight / chosen)
  };
  const DEFAULT_COLOR = '#a78bfa'; // 未知算法兜底:violet-400

  function colorFor(algo) {
    if (!algo) return DEFAULT_COLOR;
    const k = String(algo).toLowerCase();
    return COLORS[k] || DEFAULT_COLOR;
  }

  /** 当前 job 的主算法(优先 stats.algo,fallback job.algo)。 */
  function algoFor(job) {
    return (job && job.stats && job.stats.algo)
        || (job && job.algo)
        || '';
  }

  /**
   * 汇总当前 job 的所有 bbox: [{x,y,w,h,score,frame_idx,ts}, ...]
   * 视频任务可能上千帧,这里给一个上限保护避免超长 canvas 描边拖垮渲染。
   * (视觉上仍然能看清主线:按时间均匀抽样)
   */
  function collectFaces(job, maxBoxes) {
    if (!job || !job.frames) return [];
    const cap = maxBoxes || 64;
    const out = [];
    for (const f of job.frames) {
      const faces = f.faces || [];
      for (const fa of faces) {
        out.push({
          x: fa.x, y: fa.y, w: fa.w, h: fa.h,
          score: fa.score || 0,
          frame_idx: f.index,
          ts: f.timestamp_ms || 0,
        });
      }
    }
    if (out.length > cap) {
      // 等距抽样
      const step = out.length / cap;
      const sampled = [];
      for (let i = 0; i < cap; i++) sampled.push(out[Math.floor(i * step)]);
      return sampled;
    }
    return out;
  }

  /**
   * 在一个 thumbnail 容器里(已含 <img> 或 <video poster>)插入或获取 canvas overlay。
   * canvas 的实际尺寸通过 ResizeObserver 同步到 img 的渲染尺寸。
   */
  function ensureCanvas(thumbEl) {
    if (!thumbEl) return null;
    let cv = thumbEl.querySelector('canvas.sb-thumb-overlay');
    if (!cv) {
      cv = document.createElement('canvas');
      cv.className = 'sb-thumb-overlay';
      cv.setAttribute('aria-hidden', 'true');
      thumbEl.appendChild(cv);
      // 尺寸同步:跟着图片自然分辨率 + 容器渲染尺寸
      const sync = () => syncSize(thumbEl, cv);
      sync();
      if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(sync);
        ro.observe(thumbEl);
        cv._ro = ro;
      }
    }
    return cv;
  }

  function syncSize(thumbEl, cv) {
    const r = thumbEl.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width));
    const h = Math.max(1, Math.round(r.height));
    if (cv.width !== w * dpr || cv.height !== h * dpr) {
      cv.width = w * dpr;
      cv.height = h * dpr;
      cv.style.width = w + 'px';
      cv.style.height = h + 'px';
    } else {
      cv.style.width = w + 'px';
      cv.style.height = h + 'px';
    }
  }

  /**
   * 把 box 列表画到 canvas。坐标按图片 naturalSize → 容器实际显示尺寸做 object-fit:cover
   * 等比缩放(虽然 .sb-thumb 是固定 42×42 用 background-size:cover,但 <img> 是 contain:
   * 这里我们用更通用的 contain 算法以兼容两种渲染路径)。
   */
  function drawBoxes(canvas, boxes, opts) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = canvas.width / dpr, h = canvas.height / dpr;
    ctx.clearRect(0, 0, w, h);
    if (!boxes || !boxes.length) return;

    const img = canvas.parentElement && canvas.parentElement.querySelector('img');
    let natW = 0, natH = 0;
    if (img && img.naturalWidth) { natW = img.naturalWidth; natH = img.naturalHeight; }
    if (!natW) {
      // 没图(占位符 / 加载中):画一个示意框,提示"有 N 张脸"
      const c = (opts && opts.color) || DEFAULT_COLOR;
      ctx.strokeStyle = c;
      ctx.globalAlpha = 0.6;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(4, h - 14, w - 8, 10);
      ctx.fillStyle = c;
      ctx.font = '9px ui-monospace, monospace';
      ctx.fillText(`${boxes.length} 脸`, 6, h - 5);
      ctx.globalAlpha = 1;
      return;
    }

    // contain 算法
    const ar = natW / natH, sar = w / h;
    let dw, dh, dx, dy;
    if (ar > sar) { dw = w; dh = w / ar; dx = 0; dy = (h - dh) / 2; }
    else          { dh = h; dw = h * ar; dy = 0; dx = (w - dw) / 2; }
    const s = dw / natW;

    const c = (opts && opts.color) || DEFAULT_COLOR;
    ctx.lineWidth = 1;
    ctx.strokeStyle = c;
    ctx.globalAlpha = 0.85;
    ctx.shadowColor = 'rgba(0,0,0,.65)';
    ctx.shadowBlur = 1.5;
    // 单图盒子太多时只画外框,不带 score label,避免糊成一片
    const labelEvery = boxes.length > 12 ? 0 : 1;
    let drawn = 0;
    for (const b of boxes) {
      const x = dx + (b.x || 0) * s;
      const y = dy + (b.y || 0) * s;
      const bw = (b.w || 0) * s;
      const bh = (b.h || 0) * s;
      ctx.strokeRect(x, y, bw, bh);
      if (labelEvery && drawn < 4) {
        // 只在前 4 个 box 上写分数,避免遮挡
        const txt = (b.score || 0).toFixed(2);
        ctx.font = '9px ui-monospace, monospace';
        const tw = ctx.measureText(txt).width + 4;
        ctx.fillStyle = c;
        ctx.fillRect(x, y - 10, tw, 10);
        ctx.fillStyle = '#001520';
        ctx.fillText(txt, x + 2, y - 2);
      }
      drawn++;
    }
    ctx.globalAlpha = 1;
    ctx.shadowBlur = 0;
  }

  /**
   * 主入口:对指定 job 在侧栏渲染 overlay。
   * - job 为 null 或 frames 为空 → 清除
   * - job.id 与 state.currentJobId 不匹配 → 不画(节省 CPU;用户视线之外)
   */
  function refresh(job) {
    if (!job || !job.id) return;
    // 永远画当前 job + 任何 running 任务(用户可能想看到侧栏实时进度)
    const isCurrent = state.currentJobId === job.id;
    const isRunning = job.status === 'running' || job.status === 'queued';
    if (!isCurrent && !isRunning) {
      // 已结束的 job 不再画,但不清旧画布(保留最后的视觉)
      return;
    }
    const sbEl = document.getElementById('sb-' + job.id);
    if (!sbEl) return;
    const thumb = sbEl.querySelector('.sb-thumb');
    if (!thumb) return;
    const cv = ensureCanvas(thumb);
    if (!cv) return;
    const boxes = collectFaces(job);
    drawBoxes(cv, boxes, { color: colorFor(algoFor(job)) });
  }

  /** 全部清掉(切换主题 / 大批量删除时调用)。 */
  function clearAll() {
    document.querySelectorAll('.sb-thumb canvas.sb-thumb-overlay').forEach(cv => {
      const ctx = cv.getContext('2d'); if (!ctx) return;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, cv.width, cv.height);
    });
  }

  /**
   * 接入 app.js 生命周期:
   *   - mountOnRender:sidebar.renderVp 之后自动 refresh(currentJob)
   *     实现:hook sidebar.upsertJob / sidebar.setJobs
   *   - SSE:由调用方主动调用(见 app.js sse.scheduleRefresh)
   *
   * 这里用最轻的方案:init() 装一个轻量级 observer,在 DOM 变化时若发现
   * 新出现的 sb-thumb 节点就按需重画;并在 init() 时拿到 sidebar / state
   * 引用,封装一个 wrapSidebar() 把 upsertJob 包一层,自动跑 refresh。
   */
  function init() {
    if (typeof sidebar === 'undefined') return;
    // 包装 sidebar.upsertJob / setJobs — 在签名失效/数据变化时,如果涉及
    // currentJob,顺手 overlay.refresh 一次
    const origUpsert = sidebar.upsertJob;
    if (origUpsert && !origUpsert._overlay_wrapped) {
      sidebar.upsertJob = function (j) {
        const ret = origUpsert.apply(this, arguments);
        try {
          if (j && state.currentJobId === j.id) overlay.refresh({ ...state.currentJob, ...j });
          else if (j && (j.status === 'running' || j.status === 'queued')) overlay.refresh(state.jobs && state.jobs.find(x => x.id === j.id));
        } catch {}
        return ret;
      };
      sidebar.upsertJob._overlay_wrapped = true;
    }
    const origSetJobs = sidebar.setJobs;
    if (origSetJobs && !origSetJobs._overlay_wrapped) {
      sidebar.setJobs = function (list) {
        const ret = origSetJobs.apply(this, arguments);
        try {
          const cur = state.currentJob;
          if (cur) overlay.refresh(cur);
        } catch {}
        return ret;
      };
      sidebar.setJobs._overlay_wrapped = true;
    }
    // 主题切换 / 大批量重排时全部重画一次(颜色变量不变,但 canvas 被清)
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) {
        try {
          const cur = state.currentJob;
          if (cur) overlay.refresh(cur);
          else overlay.clearAll();
        } catch {}
      }
    });
  }

  return { init, refresh, clearAll, colorFor, COLORS };
})();