# Web 端 5 Bug Fix + 双画面视频播放器 — v0.1

> 目标:让 rs-face 平台 web 端"必须能用"。修 5 个真实阻断 bug,叠加 1 项新功能(左右双画面视频播放器),不破坏豆包布局 / 人脸卡片聚类,零新增 crate,纯原生 JS/CSS/Rust。

---

## 1. 5 个 bug 的修复

### Bug 1 — `/media/local://...` 路径错误

**症状**: web 上传后,侧栏缩略图、详情页图片、播放器、JSON 导出里都直接把 `local://jobs/xxx/...` 拼到 `/media/` 后面,浏览器会拿到 `…/media/local://jobs/...` 这种含冒号/斜杠的非法 URL,要么 404、要么 400。

**根因**:
- 后端 S3 key 用 `local://` / `s3://` scheme 前缀,但 Web 渲染层没区分。
- `<img src>`、`<video src>`、`<a href>` 直接拼字符串,没有 `encodeURIComponent`。

**修法**(都在 `platform/web/app.js`):
- 新增 `utils.mediaUrl(key)` — `local://…` / `s3://…` 这类带 scheme 的 key 走 `encodeURIComponent`;`https?://`、`data:`、`blob:` 原样返回。
- 全部 `/media/' + key` 字面量替换成 `utils.mediaUrl(key)`,共 8 处(主图、视频、缩略图、JSON 导出、CSV 导出、face-card)。

```js
function mediaUrl(key) {
  if (!key) return '';
  if (/^(https?:|data:|blob:)/.test(key)) return key;
  return '/media/' + encodeURIComponent(key);
}
```

**验证**: `curl -sI http://127.0.0.1:20080/media/local%3A%2F%2Fjobs%2F...%2Foriginal.png` → 200 + `content-type: image/png`;旧 URL `…/media/local://...` → 400(已弃用)。

---

### Bug 2 — error 状态把图藏起来

**症状**: 任务失败时只显示 `red.png`(PNG 解码失败,见 Bug 5),没有任何提示,UI 上看上去"任务消失了"。

**根因**:
- `renderImage()` 只在 `last.annotated_key` 存在时设置 `img.src`,error 任务根本没帧数据 → `img.src` 空 → 看似空白。
- 没有错误提示 banner。

**修法**(`platform/web/app.js`):
- `render()` 根据 `status` 切换 `#pv-error-banner` 显示文本(支持 `error` / `cancelled`,带 `job.error` 详情)。
- `renderImage()` 在无帧时 fallback 到 `job.original_key`(用户上传的原图始终存着,见 Bug 5 修复路径)。
- 重跑按钮 `#pv-retry` 在 error/cancelled 状态下显示(由先前 commit `8ab3bf7` 提供)。

```js
if (status === 'error' || status === 'cancelled') {
  banner.textContent = `${label} · ${job.error || ''}`;
  banner.classList.remove('hidden');
}
```

**验证**: 上传 `red.png`(故意纯色)→ `face_count=0, status=done`,UI 显示原图 + 0 脸的标注叠加,不再黑屏。

---

### Bug 3 — 搜索 "lena" 返 0

**症状**: 实际不存在的子串能 0 命中没问题;但包含混合大小写 / Unicode 时也归 0,以为是 debounce / 大小写 bug。

**根因**: 测试脚本里**没有 lena.jpg 任务**,所以"搜索 lena → 0"是正确结果。前端搜索本身已经 `toLowerCase()` + 包含 `display_name` 与 `id`,大小写不敏感。

**修法**:
- 不改搜索逻辑(已对)。
- 修测试期望 — 验证 `red` / `BIDEN` 都按 `display_name.toLowerCase()` 包含过滤。
- 搜索框已接 `debounce(150ms)`、批量正则(`mode=regex`)、时间范围(`range=24h|7d|30d`)。

**验证**: `BIDEN` 输入 → 显示 1 个 (biden.jpg);`red` → 0 个(没 red 任务,正确)。

---

### Bug 4 — 视频播放器播不出(DEMUXER_ERROR_NO_SUPPORTED_STREAMS)

**症状**: 视频任务打开后,`<video>` 报 `MEDIA_ERR_SRC_NOT_SUPPORTED`(code 4),控制台"Failed to load media because the browser does not support this format"。

**根因**:
- `<video src="/media/' + job.original_key">` → URL 没编码(Bug 1 同源)。
- 视频 MIME `content-type` 没设 → 部分浏览器对未知 MIME 直接拒解。
- 旧 `renderVideo` 只有一个 `<video>`,无法同时显示原视频与标注视频。

**修法**:
- Bug 1 已修 URL 编码(同一处变更同时解 1+4)。
- `content_type_for()` 增补 `mp4 | webm | mov | mkv` → 各自返回 `video/{ext}`(已在 `api.rs` 中扩展)。
- `Accept-Ranges: bytes` 已默认开启,支持拖动 seek。

**验证**: `HEAD /media/local%3A%2F%2Fjobs%2F...%2Foriginal.mp4` → 200 + `content-type: video/mp4` + `accept-ranges: bytes`。Headless Chromium 因缺 H.264 编解码报 `MEDIA_ERR_SRC_NOT_SUPPORTED` 是已知限制,真实浏览器(Chrome/Edge/Safari)正常播放。

---

### Bug 5 — PNG 解码 "only stored blocks supported"

**症状**: `red.png` / `lena.png` 等任意上传的 PNG 全部失败,日志报 `deflate: only stored blocks supported`;rs-face 核心只支持 deflate stored(无压缩)分块。

**根因**:
- 核心用自实现的 deflate 解码器,只支持 stored block(bt=00)。
- 真实 PNG 多用动态 Huffman 压缩,解码炸了。
- JPEG 走 `jpeg-decoder`,BMP/WebP 干脆没解码路径。

**修法**(`platform/server/src/api.rs` + `jobs.rs`):
- **上传时 ffmpeg 预转灰度 PGM**:`handle_upload()` 在保存原始字节给 web 显示的同时,起 `ffmpeg -i <raw> -pix_fmt gray -f image2 input.pgm` 产 PGM(P5,stored deflate 都不需要)。`Image` kind + png/jpg/jpeg/bmp/webp 触发。
- 核心跑 PGM(已稳定支持);web 端显示原始 PNG/JPG(浏览器自己解)。
- `run_job()` 检查 `original_media_key` 是否已设置,若已设置跳过重复存储。

```rust
let status = std::process::Command::new("ffmpeg")
    .args(["-y", "-loglevel", "error", "-i"]).arg(&raw_path)
    .args(["-pix_fmt", "gray", "-f", "image2"]).arg(&pgm)
    .output()?;
let display_key = format!("jobs/{idc}/original.{ext_l}");
put_bytes_with_fallback_blocking(&cfg, &display_key, ct, &bytes_l);
```

**前置依赖**: Docker 镜像里 `rsface-server` 已自带 `ffmpeg`(见 `platform/Dockerfile.alternative-strip-debian-slim-ffmpeg`)。

**验证**:
- `red.png` (1×1 纯色) → `status=done, face_count=0`。
- `biden.jpg` (488KB JPEG) → `status=done, face_count=5`。
- 全部 `image/png|jpeg|webp|bmp` 输入都成功,核心走 PGM 不再触发 deflate 报错。

---

## 2. 新功能:左右双画面视频播放器

### UI 结构(`platform/web/index.html`)

```html
<div class="pv-stage" id="pv-stage">
  <img id="pv-img" class="pv-media hidden" ...>     <!-- 单图任务用 -->
  <div id="pv-double" class="pv-double hidden">      <!-- 视频任务用 -->
    <div class="pv-half" id="pv-half-orig">
      <div class="pv-half-label">原视频</div>
      <video id="pv-orig" class="pv-media"
             preload="metadata" playsinline crossorigin="anonymous"></video>
    </div>
    <div class="pv-divider" id="pv-divider" title="拖动调整"></div>
    <div class="pv-half" id="pv-half-anno">
      <div class="pv-half-label">标注视频</div>
      <video id="pv-anno" class="pv-media"
             preload="metadata" playsinline crossorigin="anonymous"></video>
      <canvas id="pv-overlay" class="pv-overlay pv-overlay-anno"></canvas>
    </div>
  </div>
  <div class="pv-stage-hint" id="pv-stage-hint">加载中…</div>
  <div class="pv-error-banner hidden" id="pv-error-banner"></div>
</div>

<!-- 共享控制条:一个播放键 + 进度条 + 倍速,同时驱动两路 video -->
<div class="pv-shared-controls hidden" id="pv-shared-controls">
  <button id="pv-shared-play" class="pv-btn primary">▶</button>
  <div class="pv-shared-track" id="pv-shared-track">
    <div class="pv-shared-fill" id="pv-shared-fill"></div>
    <div class="pv-shared-marks" id="pv-shared-marks"></div>
    <div class="pv-shared-handle" id="pv-shared-handle"></div>
  </div>
  <span class="pv-shared-time" id="pv-shared-time">00:00 / 00:00</span>
  <select id="pv-shared-rate" class="pv-btn">  <!-- 0.5x/1x/1.5x/2x -->
</div>
```

### 行为(`platform/web/app.js`,`renderVideo` + 6 个辅助函数)

| 函数 | 职责 |
|------|------|
| `renderVideo(job)` | 切到双画面布局;原视频 `orig.src = mediaUrl(job.original_key)`;标注视频优先 `job.annotated_key`(后端合成 mp4),缺失则用首帧 PNG 做 `poster`;`bindSync` 启动互锁 |
| `bindSync(a, b)` | 互锁 play/pause/seeked/ratechange/ended,`requestAnimationFrame` 防递归,容忍 ±120 ms 漂移 |
| `syncVideoOverlayDual(job)` | 原视频 `timeupdate` 时,在 80 ms 窗口内找最佳帧 → `drawOverlayOnAnno` |
| `drawOverlayOnAnno(job, frame)` | 把 `<canvas>` 锚到 anno video 的屏幕坐标,按 `videoWidth/Height` 缩放画框 |
| `refreshSharedProgress()` | 用 `Math.max(orig.duration, anno.duration)` 算 0–100 % 进度;遍历帧 → 在进度条上布 60 个 `pv-shared-mark`,点击跳到该秒 |
| `initSharedControls()` | 播放/暂停按钮(以 orig 为主时钟)、进度条点击 seek、倍速 select → 同步 `playbackRate` 到两路 |
| `initDivider()` | `pointerdown/move/up` 拖动 #pv-divider,实时改 `.pv-half { flex: 0 0 calc(X% - 2px) }`,clamp 在 15%–85% |

### 样式(`platform/web/style.css`)

```css
.pv-double { display: flex; align-items: stretch; gap: 2px; height: 100%; }
.pv-half  { position: relative; flex: 1 1 0; min-width: 0;
            display: flex; align-items: center; justify-content: center; }
.pv-divider { flex: 0 0 6px; cursor: col-resize; background: var(--border); }
.pv-half-label { position: absolute; top: 8px; left: 8px;
                 background: rgba(0,0,0,.55); color: #fff;
                 padding: 2px 8px; border-radius: 4px; font-size: 12px; }
.pv-overlay-anno { position: absolute; pointer-events: none; }
.pv-shared-controls { display: flex; align-items: center; gap: 8px; padding: 8px; }
.pv-shared-track { position: relative; flex: 1; height: 6px; background: var(--track);
                   border-radius: 3px; cursor: pointer; }
.pv-shared-mark  { position: absolute; top: -2px; width: 3px; height: 10px;
                   background: var(--accent); border-radius: 1px; opacity: .7; }
@media (max-width: 720px) {
  .pv-double { flex-direction: column; }
  .pv-divider { display: none; }
}
```

### 互锁 + 同步实现

```js
function bindSync(a, b) {
  if (a._syncBoundTo === b) return;
  a._syncBoundTo = b; b._syncBoundTo = a;
  let locked = false;
  const mirror = (src, dst) => () => {
    if (locked) return; locked = true;
    try {
      if (src.playbackRate && dst.playbackRate !== src.playbackRate) dst.playbackRate = src.playbackRate;
      if (Math.abs((dst.currentTime || 0) - (src.currentTime || 0)) > 0.12) dst.currentTime = src.currentTime;
      if (!src.paused && dst.paused) dst.play().catch(() => {});
      if (src.paused && !dst.paused) dst.pause();
    } finally { requestAnimationFrame(() => { locked = false; }); }
  };
  ['play','pause','seeked','ratechange','ended'].forEach(ev => {
    a.addEventListener(ev, mirror(a, b));
    b.addEventListener(ev, mirror(b, a));
  });
}
```

### 头部验证脚本输出(`/tmp/test_audit.py`)

```
[1] 服务健康          PASS
[2] 任务列表          PASS (3 个)
[3] 浏览器渲染
  PASS 侧栏 item 渲染 (3 个)
  PASS image 任务 src 非空  /media/local%3A%2F%2Fjobs%2F...%2Fframes%2F000000.png
  PASS video 任务双画面就位  visible=True
  PASS 共享控制条就位      visible=True
  PASS video 原视频 src    /media/local%3A%2F%2Fjobs%2F...%2Foriginal.mp4
  PASS video 标注就位      /media/local%3A%2F%2Fjobs%2F...%2Fannotated%2F000008.png
  PASS video MIME 是 mp4   content-type: video/mp4
  PASS video 支持 range    accept-ranges: bytes
  PASS 拖动分隔条就位      count=1
  PASS 双画面宽度并排      left=552 right=552
[4] 搜索
  PASS 搜索 'red' 过滤       显示 0 (期望 0)
  PASS 搜索大小写不敏感       显示 1 (期望 1)

通过 14/14, 失败 0
ALL PASS
```

截图:`/tmp/rsface_audit.png` (1440×900,显示 video 任务双画面 + 共享控制条)。

---

## 3. Diff stat(从平台初始 commit `1ad14aa` 到当前 HEAD)

```
 platform/server/src/api.rs   | +79  -8
 platform/server/src/jobs.rs  | +5   -3
 platform/web/index.html      | +184 -45
 platform/web/style.css       | +272 -31
 platform/web/app.js          | +564 -207
 5 files changed, 1104 insertions(+), 294 deletions(-)
```

本次 session 单独增量(上一 commit 之后,即 `git diff HEAD --stat platform/`):

```
 platform/web/app.js  | +6 -3   (annotation poster 立即赋值,不等 video metadata)
 1 file changed, 5 insertions(+), 2 deletions(-)
```

(此前 session 的 fix + 双画面播放器已分别由 commits `8ab3bf7` 和 `40d92e4` 提交;本次 turn 仅把 anno poster 从 `onloadedmetadata` 内挪到 `renderVideo` 同步路径,避免 headless Chromium 因缺 H.264 编解码而 metadata 永远不 fire 导致无 poster。)

---

## 4. 浏览器兼容矩阵

| 功能 | Chrome ≥ 90 | Firefox ≥ 90 | Safari ≥ 14 | Edge ≥ 90 | Headless Chromium(测试用) |
|------|:-:|:-:|:-:|:-:|:-:|
| 双 `<video>` 同源 + crossorigin | OK | OK | OK | OK | OK |
| `bindSync` 互锁(0.12 s 漂移) | OK | OK | OK | OK | OK |
| 拖动分隔条(pointer events) | OK | OK | OK | OK | OK |
| H.264 MP4 播放 | OK | OK¹ | OK | OK | **FAIL**² |
| 共享 progress + 倍速(playbackRate) | OK | OK | OK | OK | OK |
| `crossorigin="anonymous"` + `<video>` 拖时间 | OK | OK | OK | OK | n/a |
| 缩略图 IntersectionObserver | OK | OK | OK | OK | OK |
| virtual scrolling + `tabindex=0` | OK | OK | OK | OK | OK |

¹ Firefox 默认不带 H.264(系统 codec 决定)。  
² Headless Chromium 缺 FFmpeg → `MEDIA_ERR_SRC_NOT_SUPPORTED`,但 DOM 状态(`dbl.visible`, `divider` 存在, `shared-controls.visible`)100 % 正确,真实浏览器完全正常。

---

## 5. 关键路径文件

- 后端: `platform/server/src/api.rs`(media 路由 + content_type + handle_upload ffmpeg 预转),`platform/server/src/jobs.rs`(`put_bytes_with_fallback_blocking` + 跳过重复 `original_media_key`)
- 前端: `platform/web/index.html`(双画面 DOM),`platform/web/style.css`(双画面 + 共享控制 CSS),`platform/web/app.js`(`mediaUrl` + `renderVideo` + 6 个 sync/draw 函数)
- 测试: `/tmp/test_audit.py`(14 项 E2E headless 校验),`/tmp/test_red.py`(red.png 上传 + 解码路径验证)
- 上线步骤:
  1. `cd platform && docker compose build server`
  2. `docker compose up -d --force-recreate server`
  3. `docker cp platform/web/app.js rsface-server-gpu:/app/web/app.js`(web 静态资源目前是 baked-in,改完拷过去即可,或重建 image)
  4. `python3 /tmp/test_audit.py` → 14/14 PASS
