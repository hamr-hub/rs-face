/* rs-face Platform · dropzone-preview.js
 *
 * 拖放区增强(零依赖,无构建):
 *  - 选中 / 拖入文件后,在 dropzone 下方显示预览行:
 *    缩略图(仅图)+ 文件数 + 总大小 + 头几个文件名
 *  - "✕ 清空" 按钮重置文件 input 并隐藏预览
 *  - 不改变原有"选完即上传"的自动提交行为 — 预览只是给用户一个确认机会
 *  - 新增(目录拖入):DataTransferItem.webkitGetAsEntry() 检测目录,
 *    递归 walk 目录树,筛出图片文件,返回给调用者(enqueue 到 upload-queue)
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

  // ---- 图片白名单:扩展名 + MIME ----
  const IMG_EXTS = /\.(jpe?g|png|pgm|ppm|bmp|webp|tiff?)$/i;
  function isImageFile(file) {
    if (!file) return false;
    if (file.type && /^image\//.test(file.type)) return true;
    return IMG_EXTS.test(file.name || '');
  }

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

  // ===========================================================================
  // 目录拖入支持(DataTransferItem.webkitGetAsEntry + 递归 walk)
  // 设计目标:
  //  - 单个文件:走老路径,files 列表给到调用者
  //  - 目录:递归 walk,把里面的图片文件全部收集成 [{ file, path }]
  //  - 跨浏览器:Chrome/Edge/Safari 都实现 webkitGetAsEntry;Firefox 通过
  //    DataTransfer.files 也能拿到顶层文件(只是拿不到目录结构),这种情况
  //    collectFromDataTransfer 退化为"扁平文件过滤"
  // ===========================================================================

  /**
   * 递归 walk 一个 FileSystemEntry。
   * @param {FileSystemEntry} entry
   * @param {(file: File, path: string) => void} onFile
   * @returns {Promise<void>}
   */
  function walkFolder(entry, onFile) {
    return new Promise((resolve) => {
      if (!entry) return resolve();
      if (entry.isFile) {
        entry.file(
          (file) => {
            try { onFile(file, entry.fullPath || file.name); } catch (e) {}
            resolve();
          },
          () => resolve(), // 单个文件读失败,跳过
        );
        return;
      }
      if (!entry.isDirectory) return resolve();
      // 目录:用 createReader 分批读 entries(createReader.readEntries 一次只能
      // 拿到 ~100 项,需要循环直到空)
      const reader = entry.createReader();
      let pending = 1;
      const finish = () => { if (--pending === 0) resolve(); };
      const readBatch = () => {
        reader.readEntries(
          (entries) => {
            if (!entries || !entries.length) { finish(); return; }
            pending += entries.length;
            for (const e of entries) {
              walkFolder(e, onFile).then(finish, finish);
            }
            // 浏览器实现里 readEntries 在仍有更多项时会再次调用;这里保守地
            // 再读一次直到空(浏览器会保证 readEntries 返回 [] 终止)。
            if (entries.length > 0) readBatch();
          },
          () => finish(),
        );
      };
      readBatch();
    });
  }

  /**
   * 从 DataTransfer 收集图片文件。
   * - 顶层 items 中有目录 → 递归 walk,过滤出图片
   * - 顶层 items 中只有文件 → 直接 getAsFile() + 过滤
   * - 浏览器不支持 webkitGetAsEntry → 退回 DataTransfer.files(扁平)
   * @param {DataTransfer} dt
   * @param {{accept?: (file: File) => boolean, onProgress?: (done: number, total: number) => void}} [opts]
   * @returns {Promise<{ file: File, path: string }[]>}
   */
  async function collectFromDataTransfer(dt, opts) {
    const accept = (opts && opts.accept) || isImageFile;
    const onProgress = (opts && opts.onProgress) || (() => {});
    const items = (dt && dt.items) ? Array.from(dt.items) : [];
    const files = [];
    let walked = 0;
    let walkedTotal = items.length || 0;

    if (items.length && typeof items[0].webkitGetAsEntry === 'function') {
      const promises = [];
      for (const item of items) {
        if (item.kind !== 'file') continue;
        const entry = item.webkitGetAsEntry();
        if (entry && entry.isDirectory) {
          promises.push(walkFolder(entry, (file, path) => {
            if (accept(file)) files.push({ file, path });
            walked++;
            onProgress(walked, walkedTotal);
          }));
        } else {
          const file = item.getAsFile();
          if (file) {
            if (accept(file)) files.push({ file, path: file.name });
            walked++;
            onProgress(walked, walkedTotal);
          }
        }
      }
      await Promise.all(promises);
      return files;
    }

    // Fallback:扁平 files(老浏览器或 Firefox)。DataTransfer.files 不包含
    // 目录条目(浏览器自己会跳过),所以目录拖入在这种情况下等于"用户拖进来
    // 一个看不见的容器,什么都拿不到"。这里至少处理顶层文件。
    if (dt && dt.files) {
      const arr = Array.from(dt.files);
      for (let i = 0; i < arr.length; i++) {
        const f = arr[i];
        if (accept(f)) files.push({ file: f, path: f.name });
        onProgress(i + 1, arr.length);
      }
    }
    return files;
  }

  /** 暴露的便利函数:walk 一个 entry 并只保留图片。 */
  async function walkImageFolder(entry) {
    const out = [];
    await walkFolder(entry, (file, path) => {
      if (isImageFile(file)) out.push({ file, path });
    });
    return out;
  }

  return {
    init,
    renderImage,
    renderVideo,
    clearAll,
    // 目录拖入 API(新增)
    isImageFile,
    walkFolder,
    walkImageFolder,
    collectFromDataTransfer,
  };
})();

// DOMContentLoaded 后初始化;index.html 里 defer 保证 DOM 已就绪。
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => dropzonePreview.init(), { once: true });
} else {
  dropzonePreview.init();
}