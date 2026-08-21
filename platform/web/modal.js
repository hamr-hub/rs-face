/* rs-face Platform · modal.js
 * 从 app.js 拆分(零依赖,无构建):
 *  - confirmModal: Promise 化的确认弹窗(删除/重试等危险操作)
 *  - modalKit: 通用 open/close + focus 圈闭(trap) + data-close 委托
 *  - ctxMenu: 通用右键上下文菜单(供任务卡片等使用)
 * 依赖 utils / toast(懒引用)。
 * 加载顺序:本文件必须在 app.js 之前引入。
 */
'use strict';

const modalKit = (() => {
  let _lastFocus = null;

  function open(sel) {
    const m = utils.$(sel); if (!m) return null;
    m.classList.remove('hidden');
    // a11y:记录打开前焦点,关闭后归还
    if (!_lastFocus) _lastFocus = document.activeElement;
    // 聚焦到卡片内第一个可聚焦元素
    const focusables = m.querySelectorAll('button, input, select, textarea, a[href], [tabindex]:not([tabindex="-1"])');
    if (focusables.length) { try { focusables[0].focus({ preventScroll: true }); } catch {} }
    return m;
  }

  function close(sel) {
    const m = utils.$(sel); if (!m) return;
    m.classList.add('hidden');
    if (_lastFocus && document.activeElement && m.contains(document.activeElement)) {
      try { _lastFocus.focus({ preventScroll: true }); } catch {}
      _lastFocus = null;
    }
  }

  function closeAll() {
    utils.$$('.modal').forEach(m => m.classList.add('hidden'));
    if (_lastFocus) {
      try { _lastFocus.focus({ preventScroll: true }); } catch {}
      _lastFocus = null;
    }
  }

  function anyOpen() {
    return utils.$$('.modal').some(m => !m.classList.contains('hidden'));
  }

  /** 初始化 data-close 委托(替代每个元素单独绑监听)。 */
  function init() {
    document.addEventListener('click', e => {
      const t = e.target.closest('[data-close]');
      if (!t) return;
      // 关闭最近一层 modal
      const m = t.closest('.modal');
      if (m) { m.classList.add('hidden'); if (_lastFocus) { try { _lastFocus.focus(); } catch {} _lastFocus = null; } }
    });
  }

  return { open, close, closeAll, anyOpen, init };
})();

const confirmModal = (() => {
  let _resolve = null;
  function open(title, msg, opts) {
    const m = utils.$('#modal-confirm'); if (!m) return Promise.resolve(false);
    utils.$('#modal-confirm-title').textContent = title || '确认';
    utils.$('#modal-confirm-msg').textContent = msg || '确定?';
    const ok = utils.$('#confirm-ok');
    if (ok && opts && opts.danger) { ok.textContent = opts.okText || '确定'; }
    else if (ok) { ok.textContent = opts && opts.okText ? opts.okText : '确定'; }
    m.classList.remove('hidden');
    const okBtn = utils.$('#confirm-ok'); if (okBtn) { try { okBtn.focus({ preventScroll: true }); } catch {} }
    if (_resolve) _resolve(false);
    return new Promise(res => { _resolve = res; });
  }
  function close(result) {
    const m = utils.$('#modal-confirm'); if (m) m.classList.add('hidden');
    if (_resolve) { const r = _resolve; _resolve = null; r(result); }
  }
  function init() {
    utils.$('#confirm-ok').addEventListener('click', () => close(true));
    utils.$('#confirm-cancel').addEventListener('click', () => close(false));
  }
  return { open, close, init };
})();

/**
 * 通用右键上下文菜单。
 * 用法:ctxMenu.show(e, [ {label, hint, danger, disabled, onPick}, ... ])
 * a11y:菜单 role=menu,项 role=menuitem,Esc/点击外部关闭,键盘 ↑↓ + Enter。
 */
const ctxMenu = (() => {
  let _el = null;

  function ensureEl() {
    if (_el && document.body.contains(_el)) return _el;
    _el = document.createElement('div');
    _el.className = 'ctx-menu hidden';
    _el.setAttribute('role', 'menu');
    _el.setAttribute('aria-label', '上下文菜单');
    document.body.appendChild(_el);
    // 点击外部 / Esc 关闭(挂在捕获阶段,先于业务 click)
    document.addEventListener('pointerdown', e => {
      if (_el.classList.contains('hidden')) return;
      if (!_el.contains(e.target)) hide();
    }, true);
    document.addEventListener('keydown', e => {
      if (_el.classList.contains('hidden')) return;
      if (e.key === 'Escape') { hide(); e.preventDefault(); }
      else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        const items = Array.from(_el.querySelectorAll('.ctx-item:not(.disabled)'));
        if (!items.length) return;
        const i = items.indexOf(document.activeElement);
        const n = e.key === 'ArrowDown' ? (i + 1) % items.length : (i - 1 + items.length) % items.length;
        try { items[n].focus({ preventScroll: true }); } catch {}
      }
    }, true);
    // 滚动/resize 时关闭(避免位置漂移)
    window.addEventListener('scroll', () => hide(), true);
    window.addEventListener('resize', () => hide());
    return _el;
  }

  /**
   * @param {MouseEvent} e         触发的 contextmenu 事件(会 preventDefault)
   * @param {Array}      items     [{label, hint, danger, disabled, onPick}]
   */
  function show(e, items) {
    if (!items || !items.length) return;
    const el = ensureEl();
    el.innerHTML = '';
    for (const it of items) {
      const btn = document.createElement('button');
      btn.className = 'ctx-item' + (it.danger ? ' danger' : '') + (it.disabled ? ' disabled' : '');
      btn.setAttribute('role', 'menuitem');
      btn.type = 'button';
      const label = document.createElement('span');
      label.className = 'ctx-label'; label.textContent = it.label;
      btn.appendChild(label);
      if (it.hint) {
        const hint = document.createElement('span');
        hint.className = 'ctx-hint'; hint.textContent = it.hint;
        btn.appendChild(hint);
      }
      if (it.disabled) { btn.disabled = true; }
      else btn.addEventListener('click', () => { hide(); if (it.onPick) it.onPick(); });
      el.appendChild(btn);
    }
    el.classList.remove('hidden');
    // 定位:优先在光标右下,溢出屏幕时翻转
    const vw = window.innerWidth, vh = window.innerHeight;
    const r = el.getBoundingClientRect();
    let x = e.clientX, y = e.clientY;
    if (x + r.width > vw - 8) x = Math.max(8, vw - r.width - 8);
    if (y + r.height > vh - 8) y = Math.max(8, vh - r.height - 8);
    el.style.left = x + 'px';
    el.style.top = y + 'px';
    if (e.cancelable) e.preventDefault();
    // 初始聚焦第一项(键盘可达)
    const first = el.querySelector('.ctx-item:not(.disabled)');
    if (first) { try { first.focus({ preventScroll: true }); } catch {} }
  }

  function hide() {
    if (_el) _el.classList.add('hidden');
  }

  return { show, hide };
})();
