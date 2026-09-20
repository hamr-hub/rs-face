# rs-face Platform

围绕 [rs-face](../README.md) 内核构建的人脸识别平台:
**图片 / 视频 / 直播流**识别,原始与识别结果对比,人脸时间轴,
S3(rustfs)存储,Postgres 持久化,Docker 一键部署。

```
浏览器(web/,零构建) ── REST/SSE ──▶ rsface-server(axum)
                                        │ SDK          │ SigV4
                                        ▼              ▼
                                  rs-face core      rustfs (S3)
```

## 🚀 Deploy / Start / Test → 走 Docker

**权威文档** [`DOCKER.md`](DOCKER.md) — 部署 / 启动 / 验证 / e2e / 备份 / 故障排查。

```bash
docker compose -f platform/docker-compose.yml up -d --build
curl http://localhost:20080/api/health
```

- Web 平台:<http://localhost:20080/>
- S3(rustfs):<http://localhost:19000/>
- PostgreSQL:`localhost:15432` (rsface / rsface)

数据位置:`../data/{rustfs,pg/pgdata,media}/`(bind mount,跟随代码)。

## 🛠 改 server 代码后

`platform/server/src/**` 改了之后,重新 build 镜像即可:

```bash
docker compose -f platform/docker-compose.yml up -d --build server
```

## 🎨 改前端代码后 (`platform/web/`)→ pnpm dev

后端走 Docker,前端**不要**重新 build 镜像——用 Vite dev server + HMR 即可:

```bash
docker compose -f platform/docker-compose.yml up -d --build   # 后端必须先在 :20080 跑着
pnpm install                                              # 一次性:只装 vite 作 dev server
pnpm dev                                                  # vite :5173,改 platform/web/ 下任意文件立刻 HMR
```

- vite root 是 `platform/web/`,proxy 把 `/api/*` 和 `/events` 转到 `localhost:20080`
- 编辑 `platform/web/app.js` / `index.html` / `*.css` → 浏览器立刻看到效果
- **绝对不要**直接编辑 docker 容器里的 `/app/web/`——下次 `up -d --build` 会被覆盖
- 想换后端地址:`RSFACE_BACKEND=http://host:port pnpm dev`

## 📦 GPU 部署(NVIDIA)

```bash
docker compose -f platform/docker-compose.gpu.yml up -d --build
```
需要 NVIDIA driver + nvidia-container-toolkit。CUDA 12.6 + OpenCL on Ubuntu 24.04。

## 🔗 相关文档

- 设计方案:[`docs/PLATFORM_DESIGN.md`](docs/PLATFORM_DESIGN.md)
- 路线图(AI 运维 / 算法自迭代 / 标注闭环):[`docs/ROADMAP.md`](docs/ROADMAP.md)
- core SDK 用法:[`docs/SDK.md`](docs/SDK.md)
- 镜像大小优化史(521 MB → 125 MB):[`DOCKER_SIZING.md`](DOCKER_SIZING.md)
- 最小硬件配置:[`MINIMUM_CONFIG.md`](MINIMUM_CONFIG.md)

## 配置(环境变量)

完整列表见 `server/src/config.rs`。Compose 内 `${VAR:-default}` 覆盖,或建 `platform/.env`。

| 变量 | 默认 | 说明 |
|---|---|---|
| `BIND_ADDR` | `0.0.0.0:20080` | HTTP 监听(host 网络模式直绑) |
| `S3_ENDPOINT` | `http://127.0.0.1:19000` | rustfs/S3 端点 |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | `rsface` / `rsface-secret` | S3 凭证 |
| `S3_BUCKET` | `rsface` | 桶名(不存在自动创建) |
| `RSFACE_CASCADE` | `/app/cascade.rfcf` | 级联文件路径 |
| `MAX_FRAMES_VIDEO` | `3600` | 视频任务帧数上限 |
| `MAX_FACE_CROPS` | `2000` | 每任务人脸裁剪上限 |
| `MIN_FACE_SIZE` | `24` | 最小人脸(px) |
| `STREAM_KEEPALIVE_PERIOD` | `30` | 直播流无脸帧采样周期 |

## ⚠️ 项目规则

改 compose / Dockerfile / 数据路径前,**必须同步更新 [`DOCKER.md`](DOCKER.md)**。
本目录的所有运行时都假定从 [`DOCKER.md`](DOCKER.md) 进入,不在这里重复完整步骤。
