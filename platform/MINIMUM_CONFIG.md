# rs-face Platform · 最小运行配置

> 目标:把平台跑起来需要哪些东西?一文说清。

## 1. 硬件 / 系统最低需求

| 资源 | 最低 | 推荐 |
|------|------|------|
| CPU  | 2 核 | 4 核+ |
| 内存 | 2 GB | 4 GB+(跑视频任务时) |
| 磁盘 | 5 GB(cascade + web + 镜像) | 20 GB+(保留视频原始/标注缓存) |
| 架构 | x86_64 / arm64 | — |

> 容器镜像都是 `linux/amd64` 或 `linux/arm64`,无需特殊二进制。

## 2. 软件前置依赖

| 依赖 | 用途 | 最低版本 | 安装方式 |
|------|------|----------|----------|
| Docker | 跑 rustfs / postgres / server | 20.10+ | https://docs.docker.com/engine/install/ |
| Docker Compose | 一键起三件套 | v2 (plugin) | `apt install docker-compose-plugin` |
| ffmpeg | 视频/流转码(运行在 server 容器内,镜像已自带) | 任意 | **无需宿主安装** |
| rs-face `cascade.rfcf` | Haar 分类器(已生成) | — | 仓库根自带 |

> 开发模式(不通过 Docker)还需要: Rust toolchain (1.75+)、ffmpeg 命令行、PostgreSQL 14+。

## 3. 端口规划(全部 > 10000,避免与本机常见服务冲突)

| 端口 | 用途 | 容器内端口 | 备注 |
|------|------|------------|------|
| **20080** | Web 平台 UI + REST API | 8080 | 浏览器访问 `http://localhost:20080/` |
| **19000** | rustfs S3 API | 9000 | 后端 API,无需浏览器 |
| **19001** | rustfs 控制台 | 9001 | 可选,只用于排查 |
| **15432** | PostgreSQL | 5432 | `psql -h localhost -p 15432 -U rsface` |

> 端口在 `platform/docker-compose.yml` 里改;环境变量也支持覆盖(`BIND_ADDR` 等)。

## 4. 一键启动(推荐)

```bash
cd /mnt/ssd/codespace/work/rs-face
docker compose -f platform/docker-compose.yml up -d --build
```

启动后会自动:
1. 构建 `rsface-server:latest` 镜像(多阶段,基于 `rust:1.97-slim` → `debian:trixie-slim`)。
2. 拉起 `rustfs`(S3)、`postgres`(PG 16-alpine)、`server`。
3. 等到 `rustfs` 和 `postgres` 健康后,server 才会启动。
4. server 启动时自动执行 `migrations/0001_init.sql` 建表。

**首次启动**:下载 + 编译约 5–10 分钟(取决于网络)。后续 `up -d` 跳过 build,秒级。

## 5. 验证

```bash
# 5.1 容器都 healthy
docker ps --format "table {{.Names}}\t{{.Status}}" | grep rsface

# 5.2 浏览器打开 Web 平台
xdg-open http://localhost:20080/   # 或直接访问

# 5.3 上传一张图试一下
curl -sF "file=@platform/testdata/lena.jpg" http://localhost:20080/api/jobs/image

# 5.4 看任务列表
curl -s http://localhost:20080/api/jobs | python3 -m json.tool

# 5.5 直连 PG 看持久化
docker exec -it rsface-postgres psql -U rsface -d rsface -c \
  "SELECT id, kind, status, frames_processed, total_detections FROM jobs;"
```

## 6. 数据持久化(容器卷)

| 卷名 | 内容 | 是否可删 |
|------|------|----------|
| `platform_rustfs-data` | S3 桶数据(原始视频/标注/人脸裁剪) | ⚠ 删了任务就看不到图了 |
| `platform_pg-data` | PostgreSQL job/frame/face 表 | ⚠ 删了任务历史就丢了 |
| `platform_rsface-media` | 本地媒体降级缓存(S3 失败时用) | ✅ 可随时清 |

```bash
# 想完全清空重来
docker compose -f platform/docker-compose.yml down -v
```

## 7. 仅核心(不部署)的最小配置

> 如果只想跑 `rs-face` core SDK,不需要 web/PG,直接:

```bash
cd /mnt/ssd/codespace/work/rs-face
cargo build --release
./target/release/rsface-cli --help
```

需要 1 个 `cascade.rfcf` 文件 + 任意图片/视频输入。

## 8. 故障排查(端口已被占用)

```bash
# 看谁占着
ss -tlnp | grep -E ':(20080|19000|15432) '

# 改 docker-compose.yml 里的 ports 第一段(宿主端口)
# 例:  - "28080:8080"   ← 改 28080
docker compose -f platform/docker-compose.yml up -d
```

## 9. 资源占用实测参考

| 组件 | 空闲 | 处理 1 张图 | 处理 1 个 30 秒视频 |
|------|------|------------|---------------------|
| server | 25 MB | 80 MB | 200–400 MB(单核 CPU 满载) |
| rustfs | 30 MB | 30 MB | 30 MB(只写不读) |
| postgres | 20 MB | 25 MB | 50 MB(取决于 job 数) |
| **合计** | **~75 MB** | — | **~500 MB** |

## 10. 不在最小配置里的(可后续再加)

- HTTPS / 反向代理(Nginx / Caddy):生产部署才需要
- 对象存储备份(异地):生产才需要
- 队列后端(Redis/RabbitMQ):当前 JobRegistry 已用 tokio 任务足够,见 `ROADMAP.md` v0.2
- CNN 检测器(`--cnn`):核心已实现,Web 端切换在 v0.2
- 标注/AI 自迭代:见 `ROADMAP.md` v0.3–v1.0
