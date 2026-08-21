# rs-face 平台 Web 前端

零构建、零依赖的 vanilla HTML + CSS + JavaScript 前端,服务人脸识别结果展示。

## 改动概览

仅修改 `platform/web/` 下三个文件,**不**触碰 server、Dockerfile、docker-compose、cascade.rfcf。

| 文件 | 作用 |
| --- | --- |
| `index.html` | 替换原 `.compare` 双格布局为对比 slider stage;新增热力图容器;时间轴提示文案 |
| `app.js` | 新增 `setupCompareSlider/setSlider`、`renderHeatmap`、`clusterFaces/renderClusterCard/toggleClusterExpand`;重构 `renderPanes` 配合 slider |
| `style.css` | 新增对比 slider / 热力图 / 聚类卡片样式;新增 `prefers-color-scheme: light` 主题;窄屏改上下叠放 |

## 三个核心改进

### 改进 1:对比 slider

原图(或视频)与标注图叠加在同一个 stage 内,标注层用 `clip-path: inset(...)` 实时裁剪,中间一根可拖动竖线决定左右比例。支持鼠标拖拽、点击 stage 任意位置快速跳转、键盘方向键微调、Tab 聚焦、宽 < 800px 时自动切换为水平线 + 上下叠放。

桌面 (>=800px): 左侧原图(0..pct%) | 右侧标注图(pct%..100%)  竖直分割线 + 圆形 handle
移动 (<800px):  顶部原图(0..pct%)   | 底部标注图(pct%..100%)  水平分割线 + 圆形 handle

```
+--------------------------------------------------+
|  [原图]    标注    拖动分割线 / 点击任意位置切换  |
+--------------------------------------------------+
|                  +------+                        |
|  ............... |  |...|  <- 标注图(右侧 clip)  |
|  ....原图........|  |...|                        |
|  ............... |  |...|                        |
|                  +------+                        |
|                  (<=> 句柄)                       |
+--------------------------------------------------+
```

关键代码 (app.js):
```js
function setSlider(pct) {
  state._sliderPos = Math.max(0, Math.min(100, pct));
  if (sliderAxis() === 'v') {
    line.style.top = pct + '%';
    ann.style.clipPath = `inset(${pct}% 0 0 0)`;   // 上下叠放
  } else {
    line.style.left = pct + '%';
    ann.style.clipPath = `inset(0 0 0 ${pct}%)`;  // 左右对比
  }
}
```

### 改进 2:人脸频率热力图

位于对比 slider 下方、人脸时间轴上方。把整段视频按时间段分桶(最多 140 根柱,不足则每柱 1s,过长则每柱多秒),柱高 = 该桶内人脸数。点击柱体跳转到对应时间点(视频则 `video.currentTime = ms/1000`,直播流/图片则滚动到最近的人脸卡片)。无脸的空桶用半透明灰显示,作为时间锚点。

```
+--------------------------------------------------+
|  人脸频率热力图 (403 张 / 58 柱 / 每柱 1s)        |
|  __..__..___....________________________________  |
+--------------------------------------------------+
```

关键代码 (app.js):
```js
const totalSec = Math.ceil(maxTs / 1000);
const binSize  = Math.max(1, Math.ceil(totalSec / 140));   // 限制柱数 <= 140
const counts   = new Array(numBins).fill(0);
for (const ts of faces) counts[Math.floor(ts/1000/binSize)]++;
// 每个 bar 点击 -> seekToTime(startMs)
```

### 改进 3:人脸卡片聚类

按"时间 + 空间"启发式合并连续出现的同一人脸:相邻两张人脸若 `Δt < 2000ms` 且 `bbox 中心距 < 140px` 视为同一个人。每组只显示中间帧的裁剪作为代表,角标 `xN` 提示折叠数量,点击 `v` 或角标可展开看完整时间序列(每张人脸都是 40×40 缩略,可单独点击跳转)。

合并前后对比(face-walking.mp4 实测):**403 张人脸 -> 33 组**(其中 18 组多张,最大组 93 帧)。

```
合并前(拥挤):                                    合并后(清晰):
+--+--+--+--+--+--+--+--+--+--+...   +----+-+-+--+-+-+--+-+-+...
|1 ||2 ||3 ||4 ||5 ||6 ||7 ||8 |...   |x93 ||2||28||3||16||3|...
+--+--+--+--+--+--+--+--+--+--+...   +----+-+-+--+-+-+--+-+-+...
                                              展开 x93 后:
                                              +-+-+-+-+-+-+-+-+-+
                                              | | | | | | | | | | |  <- 40x40 缩略
                                              +-+-+-+-+-+-+-+-+-+
```

关键代码 (app.js):
```js
function clusterFaces(raw) {
  const DT_MS = 2000, DIST_PX = 140;
  const clusters = []; let cur = [raw[0]];
  for (let i = 1; i < raw.length; i++) {
    const p = cur[cur.length-1], f = raw[i];
    const dt = f.ts - p.ts;
    const dist = Math.hypot((f.x+f.w/2)-(p.x+p.w/2), (f.y+f.h/2)-(p.y+p.h/2));
    if (dt < DT_MS && dist < DIST_PX) cur.push(f);
    else { clusters.push(cur); cur = [f]; }
  }
  return clusters.map(c => ({ rep: c[Math.floor(c.length/2)], members: c }));
}
```

## 主题与响应式

- **dark / light 自动切换**:`@media (prefers-color-scheme: light)` 覆盖 `--bg`/`--fg`/`--accent` 等 CSS 变量,默认仍是深色。
- **窄屏 (< 800px)**:对比 slider 改为上下叠放,分割线由竖直变水平,stage aspect-ratio 由 16/9 改为 4/5。

## 浏览器兼容性

仅用浏览器原生 API,无第三方依赖:

| 特性 | Chrome | Firefox | Safari |
| --- | --- | --- | --- |
| `clip-path: inset()` | 88+ | 54+ | 13.1+ |
| `aspect-ratio` | 88+ | 89+ | 15+ |
| `prefers-color-scheme` | 76+ | 67+ | 12.1+ |
| `EventSource` (SSE) | 6+ | 6+ | 5+ |
| `Array.prototype.flatMap` | 69+ | 62+ | 12+ |
| `Math.hypot` | 38+ | 25+ | 8+ |

三大主流浏览器(Chrome / Firefox / Safari 最新 2 个大版本)均原生支持,无需 polyfill。

## 验证步骤(已实测)

1. `docker compose -f platform/docker-compose.yml up -d` 启动
2. 浏览器打开 `http://localhost:20080/`
3. 视频识别 tab -> 上传 `platform/testdata/face-walking.mp4`
4. 等待任务完成后,工作区依次显示:
   - **对比 slider**:左侧视频(可播放)、右侧标注帧(随播放同步),拖动中间圆形 handle 即可看到左右比例实时变化
   - **热力图**:58 根柱(每柱 1s),20 根有数据;点击任一柱 -> 视频 `currentTime` 跳到对应秒
   - **人脸时间轴**:33 张卡片,其中 18 张带 `xN` 角标(最大 x93);点击角标或左下角 `v` -> 展开 93 张 40×40 缩略的完整时间序列
5. 缩小浏览器窗口至 < 800px -> slider 自动改成上下叠放(水平分割线)
6. 切换系统主题(深 / 浅)-> 整套配色自动适配

## 数据流(API)

无 server 改动,所有信息来自既有端点:

- `GET /api/jobs/{id}` -> `frames[].timestamp_ms` / `frames[].faces[].{x,y,w,h,score,key}` / `frames[].annotated_key` / `frames[].original_key` / `original_key`
- `GET /media/{key}` -> 拿图(`s3://` / `local://` 前缀由 server 解析)
- `GET /api/jobs/{id}/events` -> SSE 实时事件,直播流模式触发增量重渲染
