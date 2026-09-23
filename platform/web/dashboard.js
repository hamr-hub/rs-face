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
    const el = utils.$('#ins-stats');
    if (!el) return;
    el.innerHTML =
      `<div class="ins-section-title">概览</div>` +
      tiles.map(t =>
        `<div class="dash-tile ${t.c}"><div class="dash-tile-v">${t.v}</div><div class="dash-tile-k">${t.k}</div></div>`
      ).join('');
  }

  function renderAlgos(d) {
    const entries = Object.entries(d.algoCount).sort((a,b) => b[1]-a[1]);
    const max = Math.max(1, ...entries.map(e => e[1]));
    const el = utils.$('#ins-algos');
    if (!el) return;
    let html = `<div class="ins-section-title">算法占比</div>`;
    if (!entries.length) { html += '<div class="hint">暂无数据</div>'; el.innerHTML = html; return; }
    html += entries.map(([k,v]) => {
      const pct = (v / max * 100).toFixed(1);
      return `<div class="dash-bar-row">
        <div class="dash-bar-label">${utils.escapeHtml(k)}</div>
        <div class="dash-bar"><div class="dash-bar-fill" style="width:${pct}%"></div><span class="dash-bar-val">${v}</span></div>
      </div>`;
    }).join('');
    el.innerHTML = html;
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
    const el = utils.$('#ins-timeline');
    if (!el) return;
    el.innerHTML = `<div class="ins-section-title">最近 24h 任务数</div>` +
      `<svg viewBox="0 0 ${W} ${H + 6}" class="dash-svg" preserveAspectRatio="none">${bars}${labels}</svg>`;
  }

  async function renderTelemetry() {
    const el = utils.$('#ins-telemetry');
    if (!el) return;
    try {
      const r = await api.telemetrySummary(86400);
      if (!r || !r.enabled || !Array.isArray(r.buckets) || r.buckets.length === 0) {
        el.innerHTML = ''; el.hidden = true; return;
      }
      el.hidden = false;
      const total = r.total || 0;
      const buckets = r.buckets;
      const max = Math.max(1, ...buckets.map(b => b.count || 0));
      el.innerHTML = `<div class="ins-section-title">用户行为埋点 · 最近 24h · ${total} 条</div>` +
        buckets.map(b => {
          const pct = ((b.count || 0) / max * 100).toFixed(1);
          return `<div class="dash-bar-row">
            <div class="dash-bar-label" title="${utils.escapeHtml(b.name)}">${utils.escapeHtml(b.name)}</div>
            <div class="dash-bar"><div class="dash-bar-fill" style="width:${pct}%"></div><span class="dash-bar-val">${b.count || 0}</span></div>
          </div>`;
        }).join('');
    } catch { el.innerHTML = ''; el.hidden = true; }
  }

  async function renderLiveness() {
    const el = utils.$('#ins-liveness');
    if (!el) return;
    let groups = [];
    try {
      const r = await fetch('/api/liveness/quality-summary');
      const data = r.ok ? await r.json() : null;
      groups = (data && Array.isArray(data.groups)) ? data.groups : [];
    } catch { groups = []; }
    if (!groups.length) {
      el.innerHTML = '<div class="hint">暂无判定数据</div>';
      return;
    }
    const signals = [
      ['sharpness', '清晰度'],
      ['mean_brightness', '平均亮度'],
      ['clipped_ratio', '裁切占比'],
      ['high_freq_ratio', '高频占比'],
    ];
    const fmt = v => (v === null || v === undefined) ? '—' : Number(v).toFixed(3);
    const rows = signals.map(([key, label]) => {
      const cells = groups.map(g =>
        `<td class="dl-cell ${g.verdict === 'real' ? 'is-real' : 'is-spoof'}">${fmt(g[key])}</td>`
      ).join('');
      return `<tr><th class="dl-label">${label}</th>${cells}</tr>`;
    }).join('');
    const head = groups.map(g =>
      `<th class="dl-head ${g.verdict === 'real' ? 'is-real' : 'is-spoof'}">${g.verdict === 'real' ? '真脸' : '假脸'}<span class="dl-n">n=${g.samples}</span></th>`
    ).join('');
    el.innerHTML = `<div class="ins-section-title">活体标定 · 真脸 vs 假脸</div>` +
      `<table class="dl-table"><thead><tr><th></th>${head}</tr></thead><tbody>${rows}</tbody></table>`;
  }

  async function refresh() {
    // v0.5:渲染到右侧 Inspector 抽屉的 Stats tab(#inspector [data-ins="stats"] 面板)。
    // 抽屉 DOM 在 index.html 里已经写好 4 个 <div id="ins-*"> 锚点,这里只填内容。
    const host = document.querySelector('#inspector [data-ins="stats"]');
    if (!host) return;
    // 第一次填充 4 个锚点 div
    if (!host.querySelector('#ins-stats')) {
      host.innerHTML = `
        <div class="ins-stack">
          <div id="ins-stats"></div>
          <div id="ins-algos"></div>
          <div id="ins-timeline"></div>
          <div id="ins-telemetry" hidden></div>
          <div id="ins-liveness"></div>
          <div class="ins-meta" id="ins-meta"></div>
        </div>`;
    }
    document.querySelector('#ins-stats').innerHTML = '<div class="hint">加载中…</div>';
    try {
      const d = await compute();
      renderStats(d); renderAlgos(d); renderTimeline(d);
      const meta = document.querySelector('#ins-meta');
      if (meta) meta.textContent = `数据源:内存 + PG · 任务总数 ${d.total} · 刷新于 ${new Date().toLocaleTimeString()}`;
      renderTelemetry();
      renderLiveness();
    } catch (e) {
      const el = document.querySelector('#ins-stats');
      if (el) el.innerHTML = '<div class="hint">加载失败: ' + utils.escapeHtml(e.message) + '</div>';
    }
  }

  // 兼容旧 API:dashboard.open() 仍可被外部调用,等价于打开 Inspector + 切到 Stats。
  async function open() {
    const rs = window.__rsface;
    if (rs && rs.inspector) { rs.inspector.open(); rs.inspector.activate('stats'); }
    await refresh();
  }

  return { open, refresh };
})();
