/* rs-face Platform · visibility.js
 * 页面可见性治理:hidden 时暂停当前任务 SSE / 队列轮询;
 * 恢复时立即刷新一次 sidebar(kpi 模块已删除,不再需要绑定 timer)。
 */
'use strict';

const visibilityCtl = (() => {
  let _sseJobId = null;
  function onSseSuspend(jobId) { _sseJobId = jobId; }
  function consumeSuspendedSse() { const id = _sseJobId; _sseJobId = null; return id; }
  function isHidden() { return document.visibilityState === 'hidden'; }

  function init() {
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        if (window.__rsface && window.__rsface.state) {
          const st = window.__rsface.state;
          if (st.eventSource && st.currentJobId) {
            onSseSuspend(st.currentJobId);
            if (window.__rsface.sse) window.__rsface.sse.detach();
          }
        }
      } else if (document.visibilityState === 'visible') {
        const id = consumeSuspendedSse();
        if (id && window.__rsface && window.__rsface.sse && window.__rsface.state) {
          const st = window.__rsface.state;
          const j = st.currentJob;
          const running = j && (j.status === 'running' || j.status === 'queued');
          window.__rsface.sse.attach(id); // attach 内部先 detach;done/error 会自动 detach
          void running;
        }
        if (window.__rsface && window.__rsface.api && window.__rsface.sidebar) {
          window.__rsface.api.listJobs().then(jobs => {
            window.__rsface.sidebar.setJobs(jobs);
          }).catch(() => {});
        }
      }
    });
  }

  return { init, isHidden };
})();