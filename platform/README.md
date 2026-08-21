# rs-face Platform

围绕 [rs-face](../README.md) 内核构建的人脸识别平台:
**图片 / 视频 / 直播流**识别,原始与识别结果对比,人脸时间轴,
S3(rustfs)存储,Docker 一键部署。

```
浏览器(web/,零构建) ── REST/SSE ──▶ rsface-server(axum)
                                        │ SDK          │ SigV4
                                        ▼              ▼
                                  rs-face core      rustfs (S3)
```

- 设计方案:[docs/PLATFORM_DESIGN.md](docs/PLATFORM_DESIGN.md)
- 路线图(AI 运维 / 算法自迭代 / 标注闭环 / Client 端):[docs/ROADMAP.md](docs/ROADMAP.md)
- core SDK 用法:[docs/SDK.md](docs/SDK.md)

## 快速开始(Docker)

```bash
# 在仓库根目录(构建上下文需要 core 源码)
docker compose -f platform/docker-compose.yml up -d --build
```

- Web 平台:<http://localhost:8080/>
- S3(rustfs):<http://localhost:9000>(默认 `rsface` / `rsface-secret`)

## 快速开始(本地裸跑)

```bash
# 1) 启动 rustfs(或任意 S3 兼容服务)
docker run -d --name rustfs -p 9000:9000 \
  -e RUSTFS_ACCESS_KEY=rsface -e RUSTFS_SECRET_KEY=rsface-secret \
  -v rustfs-data:/data rustfs/rustfs

# 2) 启动 server(需要 PATH 上有 ffmpeg)
cd platform
S3_ENDPOINT=http://127.0.0.1:9000 \
S3_ACCESS_KEY=rsface S3_SECRET_KEY=rsface-secret \
RSFACE_CASCADE=../cascade.rfcf WEB_DIR=web \
cargo run --release
```

## 功能

| 入口 | 说明 |
|---|---|
| 图片识别 | 上传 PNG/JPG/BMP,左侧原图、右侧画框结果,底部人脸卡片 |
| 视频识别 | 上传文件或 URL;原视频播放,标注帧同步,时间轴人脸点击跳转 |
| 直播流 | rtsp/http 流实时检测,SSE 推送,原始帧与识别帧并排实时对比 |
| 任务历史 | 全部任务列表,点击回看 |

## 配置(环境变量)

| 变量 | 默认 | 说明 |
|---|---|---|
| `BIND_ADDR` | `0.0.0.0:8080` | HTTP 监听 |
| `S3_ENDPOINT` | `http://127.0.0.1:9000` | rustfs/S3 端点 |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | `rsface` / `rsface-secret` | S3 凭证 |
| `S3_BUCKET` | `rsface` | 桶名(不存在自动创建) |
| `RSFACE_CASCADE` | `cascade.rfcf` | 级联文件路径 |
| `MAX_FRAMES_VIDEO` | `3600` | 视频任务帧数上限 |
| `MAX_FACE_CROPS` | `2000` | 每任务人脸裁剪上限 |
| `MIN_FACE_SIZE` | `24` | 最小人脸(px) |
| `STREAM_KEEPALIVE_PERIOD` | `30` | 直播流无脸帧采样周期 |

完整列表见 `server/src/config.rs`。
