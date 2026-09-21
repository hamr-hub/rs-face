/* rs-face Platform · stats.js
 * 统计仪表板:对比 dashboard.js(平台总览)更"工程化"的视角
 *  - 总任务 / 完成 / 错误数 + 总人脸 / 总检测时长
 *  - 按算法拆分的累计检测数 + 平均 ms/帧 + 取名
 *  - Top-5 最慢任务(按 ms/帧 排序;从 stats.elapsed_ms / frames_processed 估算)
 *  - 最近 24h 任务流(每小时)+ 简单耗时分布
 *
 * 数据源:已存在的 GET /api/jobs/stats(按算法聚合) + GET /api/jobs(任务列表)。
 * 不需要新增 server endpoint,所有聚合都在客户端做。
 *
 * 依赖 utils / api / state(定义于 app.js,运行时懒引用)。
 * 加载顺序:本文件必须在 app.js 之前引入。
 */
'use strict';

const statsPanel = (() => {
  const COLOR = {
    haar:      '#4fc3f7',
    luminance: '#34d399',
    cnn:       '#f87171',
    ensemble:  '#f0c674',
  };
  function fmtMs(ms) {
    if (ms == null) return '—';
    if (ms < 1000) return Math.round(ms) + ' ms';
    if (ms < 60000) return (ms / 1000).toFixed(1) + ' s';
    return (ms / 60000).toFixed(1) + ' m';
  }

  async function compute() {
    const [jobs, algos] = await Promise.all([
      api.listJobs().catch(() => []),
      fetch('/api/jobs/stats').then(r => r.ok ? r.json() : null).catch(() => null),
    ]);
    const total = jobs.length;
    let running = 0, done = 0, err = 0, faces = 0, ms = 0, frames = 0;
    const byAlgo = {};
    for (const j of jobs) {
      if (j.status === 'running' || j.status === 'queued') running++;
      else if (j.status === 'done') done++;
      else if (j.status === 'error' || j.status === 'cancelled') err++;
      faces += j.face_count || 0;
      const st = j.stats || {};
      ms += st.elapsed_ms || 0;
      frames += st.frames_processed || 0;
      const a = (j.algo || (st.algo) || 'unknown');
      if (!byAlgo[a]) byAlgo[a] = { jobs: 0, faces: 0, ms: 0, frames: 0 };
      byAlgo[a].jobs++;
      byAlgo[a].faces += j.face_count || 0;
      byAlgo[a].ms    += st.elapsed_ms || 0;
      byAlgo[a].frames += st.frames_processed || 0;
    }
    // Top-5 slowest jobs (by ms/frame 估算,只算 frames_processed > 0 的)
    const slowest = jobs
      .filter(j => (j.stats || {}).frames_processed > 0 && (j.stats || {}).elapsed_ms > 0)
      .map(j => ({
        id: j.id,
        name: j.display_name || j.id,
        algo: j.algo || (j.stats || {}).algo || '',
        msPerFrame: ((j.stats || {}).elapsed_ms || 0) / ((j.stats || {}).frames_processed || 1),
        elapsed_ms: (j.stats || {}).elapsed_ms || 0,
        frames: (j.stats || {}).frames_processed || 0,
        faces: j.face_count || 0,
        status: j.status,
      }))
      .sort((a, b) => b.msPerFrame - a.msPerFrame)
      .slice(0, 5);
    // 24h 时间线(每小时任务数)
    const buckets = new Array(24).fill(0);
    const now = Date.now();
    for (const j of jobs) {
      if (!j.created_ms) continue;
      const h = Math.floor((now - j.created_ms) / (60 * 60 * 1000));
      if (h >= 0 && h < 24) buckets[23 - h] += 1;
    }
    return { total, running, done, err, faces, ms, frames, byAlgo, slowest, buckets, serverAlgos: algos && algos.algos };
  }

  function renderTiles(d) {
    const avgMsPerFrame = d.frames > 0 ? d.ms / d.frames : 0;
    const tiles = [
      { k: '总任务',       v: d.total,            c: 'accent' },
      { k: '进行中',       v: d.running,          c: 'warn' },
      { k: '已完成',       v: d.done,             c: 'success' },
      { k: '失败/取消',    v: d.err,             c: 'danger' },
      { k: '累计人脸',     v: d.faces,            c: 'accent' },
      { k: '累计运行时长', v: fmtMs(d.ms),        c: 'fg' },
      { k: '累计帧数',     v: d.frames,           c: 'fg' },
      { k: '平均 ms/帧',  v: avgMsPerFrame > 0 ? avgMsPerFrame.toFixed(2) + ' ms' : '—', c: 'accent' },
    ];
    utils.$('#stats-tiles').innerHTML = tiles.map(t =>
      `<div class="dash-tile ${t.c}"><div class="dash-tile-v">${t.v}</div><div class="dash-tile-k">${t.k}</div></div>`
    ).join('');
  }

  function renderAlgoTable(d) {
    const entries = Object.entries(d.byAlgo).sort((a, b) => b[1].jobs - a[1].jobs);
    const html = entries.length ? entries.map(([algo, a]) => {
      const ms = a.ms, frames = a.frames;
      const msPerFrame = frames > 0 ? (ms / frames) : 0;
      const c = COLOR[algo.toLowerCase()] || '#a78bfa';
      return `<tr>
        <td><span class="stats-algo-dot" style="background:${c}"></span>${utils.escapeHtml(algo)}</td>
        <td class="mono">${a.jobs}</td>
        <td class="mono">${a.faces}</td>
        <td class="mono">${frames.toLocaleString()}</td>
        <td class="mono">${fmtMs(ms)}</td>
        <td class="mono">${msPerFrame > 0 ? msPerFrame.toFixed(2) + ' ms' : '—'}</td>
      </tr>`;
    }).join('') : '<tr><td colspan="6" style="text-align:center;color:var(--fg-dim)">暂无数据</td></tr>';
    utils.$('#stats-algo-tbody').innerHTML = html;
    // 标题行也展示 server-side aggregate(更精确"数字")
    const serverEl = utils.$('#stats-server-meta');
    if (serverEl) {
      if (d.serverAlgos && d.serverAlgos.length) {
        const srv = d.serverAlgos.map(a => `${a.algo}:${a.detections}`).join(' · ');
        serverEl.textContent = `服务端 /api/jobs/stats: ${srv}`;
        serverEl.hidden = false;
      } else {
        serverEl.hidden = true;
      }
    }
  }

  function renderSlowest(d) {
    const list = utils.$('#stats-slowest');
    if (!list) return;
    if (!d.slowest.length) {
      list.innerHTML = '<div class="hint">暂无任务统计</div>';
      return;
    }
    const max = d.slowest[0].msPerFrame || 1;
    list.innerHTML = d.slowest.map(j => {
      const pct = Math.max(2, (j.msPerFrame / max) * 100).toFixed(1);
      const c = COLOR[(j.algo || '').toLowerCase()] || '#a78bfa';
      const msText = j.msPerFrame.toFixed(2) + ' ms/帧';
      return `<div class="stats-slow-row">
        <div class="stats-slow-name" title="${utils.escapeHtml(j.name)}">
          <button class="stats-slow-open" data-id="${utils.escapeHtml(j.id)}" title="打开此任务">${utils.escapeHtml((j.name || j.id).slice(0, 22))}</button>
        </div>
        <div class="stats-slow-algo"><span class="stats-algo-dot" style="background:${c}"></span>${utils.escapeHtml(j.algo || '·')}</div>
        <div class="stats-slow-bar"><div class="stats-slow-bar-fill" style="width:${pct}%;background:${c}"></div></div>
        <div class="stats-slow-val mono">${msText}</div>
      </div>`;
    }).join('');
    // 点击行打开任务
    list.querySelectorAll('.stats-slow-open').forEach(btn => {
      btn.addEventListener('click', () => {
        const id = btn.getAttribute('data-id');
        if (id && typeof preview !== 'undefined' && preview.open) preview.open(id);
        close();
      });
    });
  }

  function renderTimeline(d) {
    const max = Math.max(1, ...d.buckets);
    const W = 460, H = 80, bw = W / d.buckets.length;
    let bars = '', labels = '';
    d.buckets.forEach((v, i) => {
      const h = (v / max) * (H - 14);
      const x = i * bw + 1, y = H - h - 10, w = bw - 2;
      bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="1.5"><title>${i}:00 — ${v} 任务</title></rect>`;
    });
    ['现在', '-12h', '-24h'].forEach((t, i) => {
      const x = i === 0 ? W - 28 : i === 1 ? W / 2 - 12 : 0;
      labels += `<text x="${x.toFixed(0)}" y="${H + 2}">${t}</text>`;
    });
    utils.$('#stats-timeline').innerHTML =
      `<svg viewBox="0 0 ${W} ${H + 6}" class="dash-svg" preserveAspectRatio="none">${bars}${labels}</svg>`;
  }

  async function open() {
    const m = utils.$('#modal-stats');
    if (!m) return;
    m.classList.remove('hidden');
    utils.$('#stats-tiles').innerHTML = '<div class="hint">加载中…</div>';
    utils.$('#stats-algo-tbody').innerHTML = '<tr><td colspan="6" class="hint">加载中…</td></tr>';
    utils.$('#stats-slowest').innerHTML = '<div class="hint">加载中…</div>';
    try {
      const d = await compute();
      renderTiles(d);
      renderAlgoTable(d);
      renderSlowest(d);
      renderTimeline(d);
      const meta = utils.$('#stats-meta');
      if (meta) meta.textContent = `数据源 /api/jobs + /api/jobs/stats · 任务 ${d.total} · 刷新于 ${new Date().toLocaleTimeString()}`;
    } catch (e) {
      utils.$('#stats-tiles').innerHTML = '<div class="hint">加载失败: ' + utils.escapeHtml(e.message) + '</div>';
    }
  }

  function close() {
    const m = utils.$('#modal-stats'); if (m) m.classList.add('hidden');
  }

  function init() {
    // 顶栏按钮 + 关闭
    const tb = utils.$('#tb-stats');
    if (tb) tb.addEventListener('click', open);
    // modal close:由 modalKit init() 接管 data-close;这里只暴露 close() 给调用方。
  }

  return { init, open, close };
})();