/* rs-face Platform · toast.js
 * 从 app.js 拆分(零依赖,无构建):多级 toast 队列 + 折叠 pill。
 * 依赖 utils(定义于 app.js,运行时懒引用,经典 script 共享全局词法作用域)。
 * 加载顺序:本文件必须在 app.js 之前引入。
 */
'use strict';

const toast = (() => {
  const MAX_VISIBLE = 4; // 超过则折叠成 "+N more"
  const host = () => utils.$('#toast-host');
  const queue = []; // {msg, kind, expireAt}
  const livingEls = new Set(); // 当前显示的 DOM

  function ensurePill() {
    const h = host(); if (!h) return null;
    let pill = h.querySelector('.toast-pill');
    if (!pill) {
      pill = document.createElement('div');
      pill.className = 'toast-item toast-pill';
      pill.setAttribute('role', 'status');
      const msgEl = document.createElement('span'); msgEl.className = 'toast-msg pill-msg';
      const closeEl = document.createElement('button'); closeEl.className = 'toast-x'; closeEl.setAttribute('aria-label', '展开'); closeEl.textContent = '↗';
      pill.appendChild(msgEl); pill.appendChild(closeEl);
      // 默认隐藏
      pill.style.display = 'none';
      h.appendChild(pill);
      closeEl.addEventListener('click', () => {
        // 点击展开:把所有 queue 中未过期项立即显示
        const now = Date.now();
        const next = queue.filter(q => q.expireAt > now);
        queue.length = 0;
        next.forEach(q => show(q.msg, q.kind, q.expireAt - now));
        pill.style.display = 'none';
      });
    }
    return pill;
  }

  function refreshPill() {
    const pill = ensurePill(); if (!pill) return;
    const msgEl = pill.querySelector('.pill-msg');
    if (queue.length > 0) {
      msgEl.textContent = `⊕ ${queue.length} 条已折叠`;
      pill.style.display = '';
    } else {
      pill.style.display = 'none';
    }
  }

  function show(msg, kind = 'info', ms = 3500) {
    const h = host(); if (!h) return;
    // 超过可见上限 → 入队列 + 维护 pill
    if (livingEls.size >= MAX_VISIBLE) {
      queue.push({ msg, kind, expireAt: Date.now() + ms });
      refreshPill();
      return;
    }
    const el = document.createElement('div');
    el.className = 'toast-item ' + kind;
    el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
    const msgEl = document.createElement('span'); msgEl.className = 'toast-msg'; msgEl.textContent = msg;
    const closeEl = document.createElement('button'); closeEl.className = 'toast-x'; closeEl.setAttribute('aria-label', '关闭'); closeEl.textContent = '×';
    el.appendChild(msgEl); el.appendChild(closeEl);
    h.appendChild(el);
    livingEls.add(el);
    const close = () => {
      el.classList.add('leaving');
      setTimeout(() => { el.remove(); livingEls.delete(el); refreshPill(); }, 200);
    };
    closeEl.addEventListener('click', close);
    setTimeout(close, ms);
  }
  return {
    info:    (m, ms) => show(m, 'info', ms || 3000),
    success: (m, ms) => show(m, 'success', ms || 3000),
    warn:    (m, ms) => show(m, 'warn', ms || 8000),
    error:   (m, ms) => show(m, 'error', ms || 5000),
  };
})();
