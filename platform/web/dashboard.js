/* rs-face Platform · dashboard.js
 * 从 app.js 拆分(零依赖,无构建):统计仪表板(概览 tile / 算法占比 / 24h 时间线)。
 * 依赖 utils / api / state(懒引用,定义于 app.js)。
 * 加载顺序:本文件必须在 app.js 之前引入。
 */
'use strict';

const dashboard = (() => {
  async function compute() {
    const jobs = await api.listJobs();
    const total = jobs.length;
    let running = 0, done = 0, err = 0, faces = 0, ms = 0;
    const algoCount = {};
    const buckets = new Array(24).fill(0);
    const now = Date.now();
    for (const j of jobs) {
      if (j.status === 'running' || j.status === 'queued') running++;
      else if (j.status === 'done') done++;
      else if (j.status === 'error' || j.status === 'cancelled') err++;
      faces += j.face_count || 0;
      ms += (j.stats && j.stats.elapsed_ms) || 0;
      const algo = (j.stats && j.stats.algo) || (j.kind || 'image');
      algoCount[algo] = (algoCount[algo] || 0) + (j.face_count || 1);
      if (j.created_ms) {
        const hoursAgo = Math.floor((now - j.created_ms) / (60 * 60 * 1000));
        if (hoursAgo >= 0 && hoursAgo < 24) buckets[23 - hoursAgo] += 1;
      }
    }
    if (state.config && Array.isArray(state.config.available_algos)) {
      for (const a of state.config.available_algos) {
        if (!(a in algoCount)) algoCount[a] = 0;
      }
    }
    return { total, running, done, err, faces, ms, algoCount, buckets };
  }

  function renderStats(d) {
    const fmt = ms => ms < 1000 ? `${ms} ms` : ms < 60000 ? `${(ms/1000).toFixed(1)} s` : `${(ms/60000).toFixed(1)} m`;
    const tiles = [
      { k: '总任务', v: d.total, c: 'accent' },
      { k: '进行中', v: d.running, c: 'warn' },
      { k: '已完成', v: d.done, c: 'success' },
      { k: '错误', v: d.err, c: 'danger' },
      { k: '人脸总数', v: d.faces, c: 'accent' },
      { k: '总检测时长', v: fmt(d.ms), c: 'fg' },
    ];
    utils.$('#dash-stats').innerHTML = tiles.map(t =>
      `<div class="dash-tile ${t.c}"><div class="dash-tile-v">${t.v}</div><div class="dash-tile-k">${t.k}</div></div>`
    ).join('');
  }

  function renderAlgos(d) {
    const entries = Object.entries(d.algoCount).sort((a,b) => b[1]-a[1]);
    const max = Math.max(1, ...entries.map(e => e[1]));
    const el = utils.$('#dash-algos');
    if (!entries.length) { el.innerHTML = '<div class="hint">暂无数据</div>'; return; }
    el.innerHTML = entries.map(([k,v]) => {
      const pct = (v / max * 100).toFixed(1);
      return `<div class="dash-bar-row">
        <div class="dash-bar-label">${utils.escapeHtml(k)}</div>
        <div class="dash-bar"><div class="dash-bar-fill" style="width:${pct}%"></div><span class="dash-bar-val">${v}</span></div>
      </div>`;
    }).join('');
  }

  function renderTimeline(d) {
    const max = Math.max(1, ...d.buckets);
    const W = 460, H = 80, bw = W / d.buckets.length;
    let bars = '', labels = '';
    d.buckets.forEach((v, i) => {
      const h = (v / max) * (H - 14);
      const x = i * bw + 1, y = H - h - 10, w = bw - 2;
      bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="1.5"><title>${i}:00 — ${v} jobs</title></rect>`;
    });
    const lblHrs = ['现在', '-12h', '-24h'];
    const lblX = [W - 28, W/2 - 12, 0];
    lblHrs.forEach((t, i) => { labels += `<text x="${lblX[i].toFixed(0)}" y="${H + 2}">${t}</text>`; });
    utils.$('#dash-timeline').innerHTML =
      `<svg viewBox="0 0 ${W} ${H + 6}" class="dash-svg" preserveAspectRatio="none">${bars}${labels}</svg>`;
  }

  async function open() {
    utils.$('#modal-dashboard').classList.remove('hidden');
    utils.$('#dash-stats').innerHTML = '<div class="hint">加载中…</div>';
    try {
      const d = await compute();
      renderStats(d); renderAlgos(d); renderTimeline(d);
      utils.$('#dash-meta').textContent = `数据源:内存中的 /api/jobs · 任务总数 ${d.total} · 刷新于 ${new Date().toLocaleTimeString()}`;
    } catch (e) {
      utils.$('#dash-stats').innerHTML = '<div class="hint">加载失败: ' + utils.escapeHtml(e.message) + '</div>';
    }
  }

  return { open };
})();
