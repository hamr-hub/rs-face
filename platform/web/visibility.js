/* rs-face Platform · visibility.js
 * 页面可见性治理(零依赖,无构建):
 *  - document.hidden 时暂停 KPI 2s 轮询;恢复可见时立即刷新一次
 *  - hidden 时暂停当前任务 SSE;恢复时对运行中任务重新 attach
 * 由 app.js 在 kpi.init() 之后接管 kpi timer,见 visibility.bind(kpiTimer)。
 */
'use strict';

const visibilityCtl = (() => {
  let _kpiTimer = null;
  let _sseJobId = null; // hidden 时被暂停的 SSE 任务 id

  /** kpi.init() 内部创建的 interval id 交由本模块管理(先清旧防泄漏)。 */
  function bindKpiTimer(timerId) {
    if (_kpiTimer) clearInterval(_kpiTimer);
    _kpiTimer = timerId;
  }

  /** SSE 暂停/恢复钩子(app.js 的 sse 模块注入)。 */
  function onSseSuspend(jobId) { _sseJobId = jobId; }
  function consumeSuspendedSse() {
    const id = _sseJobId;
    _sseJobId = null;
    return id;
  }

  function isHidden() { return document.visibilityState === 'hidden'; }

  function init() {
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        // 暂停 KPI 轮询
        if (_kpiTimer) { clearInterval(_kpiTimer); _kpiTimer = null; }
        // 暂停 SSE:记录当前运行中任务后 detach
        if (window.__rsface && window.__rsface.state) {
          const st = window.__rsface.state;
          if (st.eventSource && st.currentJobId) {
            onSseSuspend(st.currentJobId);
            if (window.__rsface.sse) window.__rsface.sse.detach();
          }
        }
      } else if (document.visibilityState === 'visible') {
        // 恢复:KPI 立即刷新一次 + 重启 2s 轮询
        if (window.__rsface && window.__rsface.kpi && window.__rsface.kpi.refresh) {
          window.__rsface.kpi.refresh();
        }
        restartKpiTimer();
        // 恢复 SSE:重新 attach 被暂停的任务(SSE 有 last_event_id 续传,但
        // EventSource close 后需重建连接;attach 内部会先 detach)
        const id = consumeSuspendedSse();
        if (id && window.__rsface && window.__rsface.sse && window.__rsface.state) {
          const st = window.__rsface.state;
          if (st.currentJobId === id) {
            const j = st.currentJob;
            const running = j && (j.status === 'running' || j.status === 'queued');
            if (running) window.__rsface.sse.attach(id);
            else window.__rsface.sse.attach(id); // 状态未知也 attach,onmessage 里的 done/error 会自动 detach
          }
        }
        // 立即刷新一次任务列表,尽快拿到后台完成的任务状态
        if (window.__rsface && window.__rsface.api && window.__rsface.sidebar) {
          window.__rsface.api.listJobs().then(jobs => {
            window.__rsface.sidebar.setJobs(jobs);
          }).catch(() => {});
        }
      }
    });
  }

  function restartKpiTimer() {
    if (_kpiTimer) return;
    if (window.__rsface && window.__rsface.kpi && window.__rsface.kpi.refresh) {
      _kpiTimer = setInterval(() => window.__rsface.kpi.refresh(), 2000);
    }
  }

  return { init, bindKpiTimer, restartKpiTimer, isHidden, get kpiTimerId() { return _kpiTimer; } };
})();
