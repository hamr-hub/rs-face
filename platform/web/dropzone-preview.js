/* rs-face Platform · dropzone-preview.js
 *
 * 拖放区增强(零依赖,无构建):
 *  - 选中 / 拖入文件后,在 dropzone 下方显示预览行:
 *    缩略图(仅图)+ 文件数 + 总大小 + 头几个文件名
 *  - "✕ 清空" 按钮重置文件 input 并隐藏预览
 *  - 不改变原有"选完即上传"的自动提交行为 — 预览只是给用户一个确认机会
 *
 * DOM 节点(在 index.html 里已就绪):
 *   #dz-image-preview         图片 dropzone 的预览行(带 .dz-preview-thumb)
 *   #dz-video-preview         视频 dropzone 的预览行(只有图标,无缩略图)
 *
 * 加载顺序:本文件必须在 app.js 之前引入(defer 自动按顺序)。
 */
'use strict';

const dropzonePreview = (() => {
  const els = { img: null, vid: null };
  let _initialized = false;
  let _lastImgDataUrl = null;

  function init() {
    if (_initialized) return;
    _initialized = true;
    els.img = utils.$('#dz-image-preview');
    els.vid = utils.$('#dz-video-preview');
    if (els.img) wireClear(els.img, '#file-image');
    if (els.vid) wireClear(els.vid, '#file-video');
    // 模态关闭时清空预览(避免脏状态跨任务)
    document.addEventListener('click', (e) => {
      const t = e.target.closest('[data-close]');
      if (!t) return;
      clearAll();
    });
    // ESC 关闭 modal 走 modalKit.closeAll,也会触发 data-close click 委托;
    // 这里再监听 keydown 兜底(防止没经过 click 路径)
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        // 只在新建 modal 打开时清掉
        const m = utils.$('#modal-new');
        if (m && !m.classList.contains('hidden')) clearAll();
      }
    }, true);
  }

  function wireClear(previewEl, inputSel) {
    const btn = previewEl.querySelector('[data-clear]');
    if (btn) btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const input = utils.$(inputSel);
      if (input) input.value = '';
      previewEl.classList.add('hidden');
      _lastImgDataUrl = null;
      const thumb = previewEl.querySelector('.dz-preview-thumb');
      if (thumb) thumb.style.backgroundImage = '';
    });
  }

  /**
   * 渲染图片 dropzone 的预览。
   * @param {FileList|Array<File>} files
   */
  function renderImage(files) {
    init();
    if (!els.img) return;
    const arr = Array.from(files || []);
    if (!arr.length) {
      els.img.classList.add('hidden');
      return;
    }
    const totalSize = arr.reduce((a, b) => a + (b.size || 0), 0);
    const line1 = arr.length === 1
      ? arr[0].name
      : `${arr.length} 个文件 · ${humanSize(totalSize)}`;
    const names = arr.slice(0, 3).map(f => f.name);
    const line2 = arr.length > 3 ? `${names.join(', ')} 等` : names.join(', ');
    const l1 = els.img.querySelector('.dz-preview-line1'); if (l1) l1.textContent = line1;
    const l2 = els.img.querySelector('.dz-preview-line2'); if (l2) l2.textContent = line2;
    els.img.classList.remove('hidden');
    // 第一张图为图片类型时,生成 80x80 缩略图(data URL)
    const first = arr[0];
    if (first && first.type && first.type.startsWith('image/')) {
      // 仅在文件变化时重新读(避免 input change 反复触发时闪烁)
      const sig = first.name + ':' + first.size + ':' + first.lastModified;
      if (sig !== _lastImgDataUrl) {
        _lastImgDataUrl = sig;
        const reader = new FileReader();
        reader.onload = () => {
          const thumb = els.img.querySelector('.dz-preview-thumb');
          if (thumb) {
            thumb.style.backgroundImage = `url(${JSON.stringify(reader.result)})`;
          }
        };
        reader.onerror = () => {
          const thumb = els.img.querySelector('.dz-preview-thumb');
          if (thumb) thumb.style.backgroundImage = '';
        };
        reader.readAsDataURL(first);
      }
    }
  }

  /**
   * 渲染视频 dropzone 的预览(没有缩略图,只显示文件名 + 大小)。
   * @param {FileList|Array<File>} files
   */
  function renderVideo(files) {
    init();
    if (!els.vid) return;
    const arr = Array.from(files || []);
    if (!arr.length) {
      els.vid.classList.add('hidden');
      return;
    }
    const totalSize = arr.reduce((a, b) => a + (b.size || 0), 0);
    const first = arr[0];
    const line1 = arr.length === 1 ? first.name : `${arr.length} 个视频 · ${humanSize(totalSize)}`;
    const l1 = els.vid.querySelector('.dz-preview-line1'); if (l1) l1.textContent = line1;
    const l2 = els.vid.querySelector('.dz-preview-line2');
    if (l2) l2.textContent = arr.length === 1
      ? `视频 · ${humanSize(first.size || 0)} · ${first.type || '未知类型'}`
      : `${arr.slice(0, 3).map(f => f.name).join(', ')}${arr.length > 3 ? ' 等' : ''}`;
    els.vid.classList.remove('hidden');
  }

  function clearAll() {
    if (els.img) {
      els.img.classList.add('hidden');
      const thumb = els.img.querySelector('.dz-preview-thumb');
      if (thumb) thumb.style.backgroundImage = '';
    }
    if (els.vid) els.vid.classList.add('hidden');
    _lastImgDataUrl = null;
    const fi = utils.$('#file-image'); if (fi) fi.value = '';
    const fv = utils.$('#file-video'); if (fv) fv.value = '';
  }

  function humanSize(n) {
    if (n == null || isNaN(n)) return '0 B';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + ' MB';
    return (n / 1024 / 1024 / 1024).toFixed(2) + ' GB';
  }

  return { init, renderImage, renderVideo, clearAll };
})();

// DOMContentLoaded 后初始化;index.html 里 defer 保证 DOM 已就绪。
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => dropzonePreview.init(), { once: true });
} else {
  dropzonePreview.init();
}