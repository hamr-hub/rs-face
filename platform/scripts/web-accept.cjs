#!/usr/bin/env node
'use strict';
// 前端静态/交互契约验收(零外部服务):
//  1) 全部脚本可加载,无启动期错误
//  2) HTML 中 id 与 JS $('#id') 引用一致
//  3) 顶栏 / 主题 / 弹窗 / 搜索 关键交互可执行
//
// 运行(在仓库根目录):
//   node platform/scripts/web-accept.cjs
// 依赖:仅 devDependency jsdom(pnpm install 后位于根 node_modules)。
//
// 适配 DESIGN.md 重设计后的 UI(2026-09-30):
//   - 顶栏 3 元素(logo+title / search / theme),无 #tb-new/#tb-settings/#tb-help
//   - 侧栏 1 列,3 段时间分组,无 .kpi / .sb-filter / .new-type / .sb-algo-chip
//   - 主区三态 empty/detail,无 Inspector / dashboard / settings modal
const { JSDOM, VirtualConsole } = require('jsdom');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..', '..');
const WEB = path.join(ROOT, 'platform', 'web');

const html = fs.readFileSync(path.join(WEB, 'index.html'), 'utf8');

(async () => {
  const errors = [];
  const vc = new VirtualConsole();
  vc.on('jsdomError', (e) => errors.push(['jsdomError', e.message]));
  vc.on('error', (...a) => errors.push(['console.error', a.join(' ')]));

  const dom = new JSDOM(html, {
    url: 'http://rsface.local/',
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    virtualConsole: vc,
  });
  const { window } = dom;

  window.matchMedia = window.matchMedia || (() => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {} }));
  window.ResizeObserver = class { observe(){} unobserve(){} disconnect(){} };
  window.IntersectionObserver = class { observe(){} unobserve(){} disconnect(){} takeRecords(){return [];} rootBounds=null; };
  window.EventSource = class { constructor(){ this.readyState = 0; } close(){} addEventListener(){} };
  window.HTMLCanvasElement.prototype.getContext = window.HTMLCanvasElement.prototype.getContext || (() => null);
  let store = {};
  Object.defineProperty(window, 'localStorage', {
    value: {
      getItem: k => (k in store ? store[k] : null),
      setItem: (k, v) => { store[k] = String(v); },
      removeItem: k => { delete store[k]; },
      clear: () => { store = {}; },
    },
    configurable: true,
  });
  // 无后端时 fetch 静默空响应
  window.fetch = async () => ({
    ok: false, status: 0,
    json: async () => ({}), text: async () => '',
  });

  // 加载顺序与 index.html 一致:utils → toast → modal → visibility →
  // upload-queue → dropzone-preview → app(主模块,依赖前面所有)。
  const scripts = [
    'utils.js', 'toast.js', 'modal.js', 'visibility.js',
    'upload-queue.js', 'dropzone-preview.js', 'app.js',
  ];
  // 经典 <script> 顶层 const/let 共享同一个脚本词法作用域,拼接执行还原之
  const bundle = scripts
    .map(f => fs.readFileSync(path.join(WEB, f), 'utf8') + `\n//# sourceURL=${f}`)
    .join('\n;\n');
  try {
    new window.Function(bundle).call(window);
  } catch (e) {
    errors.push(['script:bundle', e.message]);
  }

  await new Promise(r => setTimeout(r, 600));

  const doc = window.document;

  // ---- JS 引用的 id 必须在 HTML 中存在(可选面板除外:upload-queue.js / dropzone-preview.js
  // 的 init() 设计成"元素不在就静默 no-op",DESIGN.md 减法优先允许它们没 DOM 容器) ----
  const OPTIONAL_SCRIPTS = new Set(['upload-queue.js', 'dropzone-preview.js']);
  const htmlIds = new Set([...doc.querySelectorAll('[id]')].map(e => e.id));
  const missingIds = [];
  for (const f of scripts) {
    if (OPTIONAL_SCRIPTS.has(f)) continue;
    const code = fs.readFileSync(path.join(WEB, f), 'utf8');
    const re = /\$\(['"]#([\w-]+)['"]\)/g;
    let m;
    while ((m = re.exec(code))) {
      if (!htmlIds.has(m[1])) missingIds.push(`${f}: #${m[1]}`);
    }
  }
  if (missingIds.length) errors.push(['missing-id', missingIds.join('; ')]);

  const checks = [];
  const pending = [];
  function check(name, fn) {
    pending.push((async () => {
      try {
        const r = await fn();
        checks.push([name, r !== false ? 'PASS' : 'FAIL']);
      } catch (e) {
        checks.push([name, 'ERROR', e.message]);
      }
    })());
  }

  check('顶栏元素齐全(tb-drawer/tb-home/tb-theme)', () => {
    return ['#tb-drawer', '#tb-home', '#tb-theme', '#tb-search', '#tb-search-clear'].every(s => doc.querySelector(s));
  });
  check('顶栏无旧图标(已删 tb-new/tb-settings/tb-help)', () => {
    return !['tb-new', 'tb-settings', 'tb-help'].some(id => doc.getElementById(id));
  });
  check('侧栏 3 段容器存在(sb-scroll/vp/spacer)', () => {
    return ['#sb-scroll', '#sb-vp', '#sb-spacer'].every(s => doc.querySelector(s));
  });
  check('主区三态容器(empty/detail/preview)', () => {
    return ['#pv-empty', '#pv-detail', '#preview'].every(s => doc.querySelector(s));
  });
  check('空态引导文案居中', () => {
    return doc.querySelector('#pv-empty h2')?.textContent?.includes('拖入');
  });
  check('主题属性可切 dark/light', () => {
    doc.documentElement.setAttribute('data-theme', 'dark');
    const d = doc.documentElement.getAttribute('data-theme') === 'dark';
    doc.documentElement.setAttribute('data-theme', 'light');
    return d && doc.documentElement.getAttribute('data-theme') === 'light';
  });
  check('主题按钮可点击(auto→dark→light→auto 循环)', () => {
    const btn = doc.getElementById('tb-theme');
    if (!btn) return false;
    const start = btn.dataset?.mode || btn.getAttribute('data-mode');
    btn.click();
    const after = btn.dataset?.mode || btn.getAttribute('data-mode');
    return start !== after;
  });
  check('搜索输入可写 + 清空按钮可见性', () => {
    const inp = doc.getElementById('tb-search');
    const clr = doc.getElementById('tb-search-clear');
    if (!inp || !clr) return false;
    inp.value = 'test';
    inp.dispatchEvent(new window.Event('input', { bubbles: true }));
    // onSearch 是 debounce 80ms,等一拍再读
    return (async () => {
      await new Promise(r => setTimeout(r, 120));
      const wasVisible = !clr.classList.contains('hidden');
      clr.click();
      const cleared = inp.value === '' && clr.classList.contains('hidden');
      return wasVisible && cleared;
    })();
  });
  check('新建 modal 开/关', () => {
    doc.querySelector('#pv-empty-new').click();
    const opened = !doc.querySelector('#modal-new').classList.contains('hidden');
    doc.querySelector('#modal-new .modal-x').click();
    return opened && doc.querySelector('#modal-new').classList.contains('hidden');
  });
  check('confirm modal 可开/关', () => {
    doc.querySelector('#modal-confirm').classList.remove('hidden');
    const opened = !doc.querySelector('#modal-confirm').classList.contains('hidden');
    doc.querySelector('#modal-confirm .modal-x').click();
    return opened && doc.querySelector('#modal-confirm').classList.contains('hidden');
  });
  check('dropzone-overlay 全局拖入容器存在', () => !!doc.querySelector('#dropzone-overlay'));
  check('runtime 徽章初始隐藏(待 loadConfig 显示)', () => {
    const el = doc.getElementById('tb-runtime');
    return el && el.hidden === true;
  });
  check('status-bar 快捷键文案含 n/j/k/⌫/?', () => {
    const t = doc.getElementById('status-bar')?.textContent || '';
    return /n 新建/.test(t) && /j\/k/.test(t) && /删除/.test(t);
  });
  check('favicon 声明为 svg', () => {
    return [...doc.querySelectorAll('link[rel=icon]')].some(l => /svg/.test(l.type || ''));
  });
  check('帮助 modal + 状态栏触发按钮存在', () => {
    const m = doc.querySelector('#modal-help');
    const btn = doc.querySelector('#sb-help');
    if (!m || !btn) return false;
    // kbd 列表至少 8 项 + 三段 help-h3 + 列表
    const kbd = m.querySelectorAll('kbd').length;
    const h3 = m.querySelectorAll('.help-h3').length;
    return kbd >= 8 && h3 >= 3;
  });
  check('帮助按钮可点击开关', () => {
    const btn = doc.querySelector('#sb-help');
    const m = doc.querySelector('#modal-help');
    if (!btn || !m) return false;
    const before = m.classList.contains('hidden');
    btn.click();
    const opened = !m.classList.contains('hidden');
    btn.click();
    const closed = m.classList.contains('hidden');
    return before && opened && closed;
  });
  check('算法入口是 button 不是 anchor (可键盘聚焦)', () => {
    const btn = doc.querySelector('#algo-link');
    if (!btn) return false;
    if (btn.tagName !== 'BUTTON') return false;
    if (btn.getAttribute('aria-expanded') == null) return false;
    return true;
  });
  check('算法面板 ≥ 6 个 chip 且"默认"始终在首位', () => {
    const panel = doc.querySelector('#algo-panel');
    if (!panel) return false;
    const chips = panel.querySelectorAll('.algo-chip');
    if (chips.length < 6) return false;
    const first = chips[0];
    return first && first.dataset.algo === '' && first.textContent.includes('默认');
  });
  check('算法选择持久化 (localStorage + __algoPicker)', () => {
    const p = doc.defaultView;
    if (!p || !p.__algoPicker || typeof p.__algoPicker.get !== 'function') return false;
    const before = p.__algoPicker.get();
    p.__algoPicker.select('cnn');
    const after = p.__algoPicker.get();
    const stored = p.localStorage.getItem('rsface.algo');
    p.__algoPicker.select(before || '');
    return after === 'cnn' && stored === 'cnn';
  });

  await Promise.all(pending);

  const realErrors = errors.filter(([, msg]) => !/Not implemented|ResizeObserver|IntersectionObserver/.test(msg || ''));
  console.log('=== 前端验收 ===');
  let fail = 0;
  for (const [n, s, extra] of checks) {
    console.log(`${s === 'PASS' ? '✓' : '✗'} ${n}${s !== 'PASS' && extra ? ' — ' + extra : ''}`);
    if (s !== 'PASS') fail++;
  }
  console.log('\n=== 运行时错误 ===');
  if (!realErrors.length) console.log('(none)');
  for (const e of realErrors) { console.log('✗', e.join(' | ')); fail++; }

  process.exit(fail ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });