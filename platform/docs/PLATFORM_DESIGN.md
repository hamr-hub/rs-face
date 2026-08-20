# rs-face Platform 设计方案(v0.1)

> 本文档是平台层的总体方案。识别内核(core)的设计见仓库根 `docs/architecture.md`,
> 本文档只描述"围绕 core 构建的工程化平台"。

## 1. 背景与定位

rs-face core 是一个零依赖、纯 Rust 的 Viola-Jones(+CNN)人脸检测内核。
平台的定位不是"一个演示 Demo",而是一个**可长期迭代的开源人脸识别工程体系**:

- **core 即 SDK**:识别能力以 `rs-face` crate 形式对外输出,任何 Rust 项目可直接依赖;
- **Web 管理端**只是平台的一个入口,未来还有 Client 端(CLI/桌面/边缘设备);
- **数据闭环**:识别结果、错误标注、算法版本、评测集都沉淀为资产,支撑
  **AI 运维 / 算法自迭代 / 结果可视化 / 优化前后效果比对**;
- 最终收敛于一个**高效、性能最强、效果最好**的人脸识别开源工具:核心套件 + 生态外围。

### 分层原则(本项目最重要的约束)

```
┌─────────────────────────────────────────────────────────┐
│  应用层:  Web 管理端 │ 未来 Client 端 │ 未来标注/评测工具  │
├─────────────────────────────────────────────────────────┤
│  平台层:  platform/ (本目录,独立 Cargo 包)               │
│           · 任务引擎 · REST/SSE API · S3 存储 · 部署       │
├─────────────────────────────────────────────────────────┤
│  内核层:  src/ (rs-face core,零依赖,不感知平台存在)      │
└─────────────────────────────────────────────────────────┘
```

- **core 不 import 平台任何代码**,保持零依赖与可独立发布;
- 平台通过 `rsface = { package = "rs-face", path = ".." }` 把 core 当 SDK 使用;
- 平台目录 `platform/` 与 core 源码 `src/` 完全隔离,各自独立编译。

## 2. 总体架构(v0.1 已实现)

```text
                       ┌────────────────────────────┐
   浏览器 ──HTTP/SSE──▶│  rsface-platform server    │
   (web/, 纯静态)       │  · axum REST API            │
                       │  · 任务引擎(std 线程)       │
                       │  · SSE 实时事件              │
                       │  · S3 客户端(SigV4,自研)    │
                       └──────┬───────────┬─────────┘
                              │ SDK 调用   │ S3 REST
                              ▼           ▼
                     ┌──────────────┐  ┌──────────────────┐
                     │ rs-face core │  │ rustfs (S3 兼容)  │
                     │ 级联/检测器/  │  │ 原始媒体/标注帧/  │
                     │ 帧源(ffmpeg)│  │ 人脸裁剪/权重     │
                     └──────────────┘  └──────────────────┘
```

组件职责:

| 组件 | 技术 | 职责 |
|---|---|---|
| core | 零依赖 Rust | 级联加载、检测、帧源(PNG 序列/HTTP/ffmpeg pipe/合成流)、PNG 编解码 |
| server | axum + tokio + 自研 SigV4 S3 客户端 | 任务调度、REST/SSE、媒体代理、静态托管 |
| web | vanilla HTML/JS/CSS,零构建 | 上传/URL 发起识别、原始 vs 结果对比、人脸时间轴 |
| rustfs | Docker 部署的 S3 兼容存储 | 全部媒体与未来数据资产(标注、权重、评测集)的持久层 |

## 3. 目录结构

```
rs-face/                    # core(不动)
├── src/…                   # 内核:haar/detector/pipeline/source/output/cnn/gpu
├── cascade.rfcf            # OpenCV frontalface 级联(rfcf 格式)
└── platform/               # ★ 平台层(本方案的主体,独立 Cargo 包)
    ├── Cargo.toml          # rsface-platform;依赖 core(path)+ axum/tokio/ureq/sha2/hmac
    ├── server/src/
    │   ├── main.rs         # 入口:装配配置/S3/路由
    │   ├── config.rs       # 环境变量配置(全部可覆盖)
    │   ├── s3.rs           # 极简 S3 客户端(AWS SigV4;PUT/GET/建桶)
    │   ├── jobs.rs         # 任务引擎:检测循环、标注帧、人脸裁剪、S3 上传、事件
    │   └── api.rs          # REST + SSE + 媒体代理 + 静态托管
    ├── web/                # 前端(index.html / app.js / style.css,零构建)
    ├── docs/               # PLATFORM_DESIGN.md / ROADMAP.md / SDK.md
    ├── Dockerfile          # 多阶段构建(rust 构建 + ffmpeg 运行时)
    ├── docker-compose.yml  # rustfs + rsface-server 一键部署
    └── README.md
```

## 4. 任务模型与数据流

### 4.1 任务类型

| kind | 输入 | 帧来源 | 生命周期 |
|---|---|---|---|
| `image` | 上传图片(PNG/JPG/BMP…,非 PNG 由 ffmpeg 归一化) | 单帧 | 一次性 |
| `video` | 上传视频文件 或 URL(mp4/rtsp/http…) | ffmpeg pipe | 一次性(帧数上限保护) |
| `stream` | 流地址(rtsp://… 等) | ffmpeg pipe | 持续,直到取消/流结束 |

### 4.2 状态机

```
queued ──▶ running ──▶ done
              │  ▲
              │  └── (取消信号)
              └──▶ cancelled
              └──▶ error(cascade 缺失/源打不开等)
```

### 4.3 检测循环(jobs.rs)

```
打开帧源 ─▶ 循环 { 取帧 → 检测 → 有脸?/keepalive?
                    → 生成 RGB 底图 → 画框(标注帧)→ 裁剪人脸
                    → PNG 编码 → S3 put → 追加 FrameResult → SSE 推送 }
        ─▶ 终态(done/cancelled/error)→ 统计落库 → SSE 终态事件
```

- 直播流每 N 帧(`STREAM_KEEPALIVE_PERIOD`,默认 30)存一对"原始帧 + 标注帧",
  保证无脸时画面仍然活动,且**原始与识别结果始终可对比**;
- 人脸裁剪数量上限 `MAX_FACE_CROPS` 防止长流撑爆存储;
- 视频任务帧数上限 `MAX_FRAMES_VIDEO`(默认 3600)。

### 4.4 S3 对象布局(rustfs)

```
jobs/{job_id}/original.{ext}        # 原始图片/视频
jobs/{job_id}/frames/{index}.png    # 原始帧(图片任务/直播流)
jobs/{job_id}/annotated/{index}.png # 画框标注帧
jobs/{job_id}/faces/{index}_{k}.png # 人脸裁剪图(时间轴卡片)
```

> 预留:`datasets/`(标注数据)、`models/`(权重版本)、`benchmarks/`(评测结果)——见 ROADMAP。

## 5. API 规范

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| POST | `/api/jobs/image` | multipart `file` 字段;返回 `{job_id}` |
| POST | `/api/jobs/video` | multipart `file` 字段;返回 `{job_id}` |
| POST | `/api/jobs/stream` | JSON `{url}`;返回 `{job_id}` |
| GET | `/api/jobs` | 任务列表(摘要) |
| GET | `/api/jobs/{id}` | 任务详情:状态、统计、`frames[]`(含 `annotated_key`/`original_key`/`faces[]`) |
| POST | `/api/jobs/{id}/cancel` | 取消运行中的任务(直播流停止) |
| GET | `/api/jobs/{id}/events` | SSE:`{"type":"frame"|"done"|"cancelled"|"error",…}` |
| GET | `/media/{key}` | S3 媒体代理(带缓存头;前端无需感知 S3 凭证) |

`FrameResult` 结构:

```json
{
  "index": 137,
  "timestamp_ms": 4567,
  "annotated_key": "jobs/abc/annotated/000137.png",
  "original_key": null,
  "faces": [
    {"key": "jobs/abc/faces/000137_0.png",
     "x": 210, "y": 96, "w": 88, "h": 88, "score": 3.42}
  ]
}
```

## 6. 前端交互设计

- **三个入口**:图片(拖拽/点选上传)、视频(文件或 URL)、直播流(URL + 停止按钮);
- **对比视图**:左"原始"右"识别结果"并排;图片任务为双图对比;
  视频任务左侧为可播放原视频,右侧标注帧随 `timeupdate` 同步到最近的已识别帧;
  直播流左右分别为最新原始帧与标注帧;
- **人脸时间轴**(底部横条):按出现时间先后排序的人脸裁剪卡片,
  显示时间点(`mm:ss.d`)与置信度;**点击卡片 → 视频跳转到对应时间点**;
- **任务历史**:列出全部任务(状态/帧数/人脸数),点击回看;
- 实现为零构建 vanilla JS,便于嵌入任意静态托管。

## 7. 部署(Docker)

```yaml
services:
  rustfs:         # S3 兼容存储,Docker Hub rustfs/rustfs
  rsface-server:  # 多阶段构建:rust:1.97 编译 + 运行时含 ffmpeg
                   # 环境变量注入 S3 端点/凭证/桶名/级联路径
```

`docker compose up -d` 后:

- Web 平台:`http://<host>:8080/`
- S3(rustfs):`http://<host>:9000`

构建上下文为仓库根(需要 core 源码参与编译),`.dockerignore` 排除 `target/`、`.git/`。

## 8. 关键设计决策

| 决策 | 理由 |
|---|---|
| core 保持零依赖、平台独立成包 | 内核可独立发布为 SDK;平台演进不污染内核 |
| 自研 SigV4 S3 客户端(ureq + sha2/hmac,无 TLS) | 内网访问 rustfs,避免引入 aws-sdk 的庞大依赖树;~200 行可控代码 |
| 检测循环直接用 `Detector` 而非 `Pipeline` | 平台需要逐帧产出人脸裁剪 + SSE 事件,Pipeline 的"写目录+manifest"模型不匹配 |
| 媒体一律过 `/media/` 代理 | 前端零凭证、零 CORS 配置;未来可换 CDN/预签名 |
| 前端零构建(vanilla JS) | 部署即拷贝;后续如复杂化可平滑迁移 Vite 等 |
| 任务状态先放内存 | v0.1 单实例;v0.2 引 SQLite/PG 持久化(ROADMAP) |

## 9. 已知限制(v0.1)

- 任务元数据仅存内存,服务重启即失(媒体仍在 S3);
- MJPEG 直接输入未支持(core 零依赖限制,统一走 ffmpeg);
- 单实例;检测线程为每任务单线程(core 的多线程 Pipeline 未接入平台);
- 无鉴权(内网部署假设),公网部署需自加反代鉴权。

## 10. 与长期愿景的衔接

平台从 v0.1 起就把**所有数据资产放 S3**,这是为闭环做的最重要的铺垫:
识别结果(`jobs/`)之后将扩展 `datasets/`(错误标注)、`models/`(权重版本)、
`benchmarks/`(评测报告),使"标注 → 重训 → 评测 → 前后效果比对"成为平台内建的
数据流而非外部脚本。详见 [ROADMAP.md](ROADMAP.md)。
