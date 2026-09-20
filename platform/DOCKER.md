# rs-face Platform — Docker 部署与运维

> 这是 `platform/` 子系统的**权威运维文档**。改 compose / Dockerfile / 数据路径前,
> 请同步更新本页。规则来源:[`../CLAUDE.md`](../CLAUDE.md)。

## TL;DR

```bash
docker compose -f platform/docker-compose.yml up -d --build

# 验证
curl http://localhost:20080/api/health
# → {"service":"rsface-platform","status":"ok"}
```

启动后:
- Web 平台:<http://localhost:20080/>
- S3(rustfs):<http://localhost:19000/>
- S3 控制台:<http://localhost:19001/>
- PostgreSQL:`localhost:15432` (rsface / rsface-pass / db=rsface)

## 服务清单

| 服务 | 镜像 | 端口(HOST:CONT) | bind mount | 角色 |
|---|---|---|---|---|
| `rsface-server` | 本地 build(`platform/Dockerfile`,arm64-musl + alpine,~125 MB) | `20080`(host 网络模式) | `../data/media` → `/var/rsface-media` | Web + REST API + SSE |
| `rsface-postgres` | `postgres:16-alpine` | `15432:5432` | `../data/pg/pgdata` → `/var/lib/postgresql/data` | 任务历史(jobs/frames/faces) |
| `rsface-rustfs` | `rustfs/rustfs:latest` | `19000:9000` + `19001:9001` | `../data/rustfs` → `/data` | S3 对象存储(媒体/结果) |

`rsface-server` 用 `network_mode: host` 直接绑 host `0.0.0.0:20080`,与另外两个容器
通过 `127.0.0.1:<host-port>` 通信;**docker iptables NAT 完全绕开**,端口转发是 1:1 直绑。

## 启动 / 停止 / 查看

| 操作 | 命令 |
|---|---|
| 启动(首次会 build 镜像,5–10 分钟;之后秒起) | `docker compose -f platform/docker-compose.yml up -d --build` |
| 启动(已 build,跳过镜像重建) | `docker compose -f platform/docker-compose.yml up -d` |
| 状态 + 健康 | `docker compose -f platform/docker-compose.yml ps` |
| 实时日志(3 服务) | `docker compose -f platform/docker-compose.yml logs -f --tail=100` |
| 单服务日志 | `docker logs -f rsface-server` |
| 优雅停止(SIGTERM 等 2m30s) | `docker compose -f platform/docker-compose.yml down` |
| 强制停止(立即 SIGKILL) | `docker compose -f platform/docker-compose.yml kill rsface-server` |
| **完全清理(数据会丢,慎用)** | `docker compose -f platform/docker-compose.yml down && rm -rf data/rustfs data/pg/pgdata data/media` |

## 验证服务健康

```bash
# 1. rsface-server
curl -sf http://localhost:20080/api/health
# → {"service":"rsface-platform","status":"ok"}

# 2. rustfs
curl -sf http://localhost:19000/minio/health/live
# → {"status":"ok","ready":true,"service":"rustfs-endpoint",...}

# 3. postgres
docker exec rsface-postgres pg_isready -U rsface -d rsface
# → localhost:15432 - accepting connections

# 4. 查看表(确认 schema + 数据)
docker exec rsface-postgres psql -U rsface -d rsface -c "\dt"
docker exec rsface-postgres psql -U rsface -d rsface -c "SELECT count(*) FROM jobs;"
```

如果 4 个检查都通过,整个栈是健康的。

## 端到端测试 (e2e)

### 一键 smoke

```bash
bash platform/scripts/docker-smoke.sh
```

会跑:`/api/health` → `/api/jobs/image` 上传 → PG jobs 计数。

### 手动 API 测试

```bash
# 上传图片
curl -F file=@/path/to/face.jpg \
     http://localhost:20080/api/jobs/image

# 流式订阅任务进度
curl -N "http://localhost:20080/events?last_event_id="

# 查任务状态
curl http://localhost:20080/api/jobs/<job_id>

# 取消任务
curl -X POST http://localhost:20080/api/jobs/<job_id>/cancel
```

### 并发压测(对运行中的栈)

```bash
bash platform/testdata/scripts/load_test.sh
# 100 并发 /api/jobs/image,打印 p50/p95/p99
```

## PG 备份与还原

### 备份(导出当前 PG)

```bash
docker exec rsface-postgres pg_dump \
    -U rsface -d rsface -F c \
    -f /tmp/rsface_dump.sqlc

docker cp rsface-postgres:/tmp/rsface_dump.sqlc \
    data/pg/rsface_dump.sqlc
```

dump 是 postgres 自定义格式(-F c),压缩、二进制,适合 `pg_restore`。

### 还原(初始化或迁移)

**前置条件**:目标栈的 `data/pg/pgdata/` 是空目录(或确认要覆盖的库)。
**`pgdata` 非空** 时 `pg_restore --clean` 会先把现有 jobs/frames/faces 清掉再灌。

```bash
docker exec -i rsface-postgres pg_restore \
    -U rsface -d rsface --clean --if-exists --no-owner --role=rsface \
    < data/pg/rsface_dump.sqlc
```

如果 dump 是空表(刚 initdb 的空库),`--clean` 会跳过 missing objects,不报错。

## 数据迁移(从旧的 named volume)

如果你的部署之前用 docker named volume(`platform_rsface-media` /
`platform_pg-data` / `platform_rustfs-data`),迁移到 bind mount 的步骤:

```bash
# 1. 从旧 PG 容器导 dump
docker exec rsface-postgres pg_dump -U rsface -d rsface -F c -f /tmp/dump.sqlc
docker cp rsface-postgres:/tmp/dump.sqlc data/pg/rsface_dump.sqlc

# 2. 从旧 rsface 容器拷媒体(7.1G 量级,需要时间)
docker cp rsface-server:/var/rsface-media/. data/media/

# 3. 从旧 rustfs 容器拷对象(160K)
docker cp rsface-rustfs:/data/. data/rustfs/

# 4. 停旧栈,删旧 named volume
docker compose down
docker volume rm platform_rsface-media platform_pg-data platform_rustfs-data

# 5. 重启新栈(用 bind mount)+ 还原 PG
docker compose -f platform/docker-compose.yml up -d --build
docker exec -i rsface-postgres pg_restore \
    -U rsface -d rsface --clean --if-exists --no-owner --role=rsface \
    < data/pg/rsface_dump.sqlc
```

完成后 `data/` 目录结构:
```
data/
├── media/              # 上传媒体 + 检测结果(7.1G 量级)
├── pg/
│   ├── pgdata/         # postgres 数据(自动 initdb,非空目录)
│   └── rsface_dump.sqlc  # 备份 dump(可删,但留着方便回滚)
└── rustfs/             # S3 对象(160K 量级)
```

## 故障排查

| 症状 | 检查 | 修复 |
|---|---|---|
| `rsface-server` 显示 `unhealthy` | `docker inspect rsface-server --format '{{json .State.Health.Log}}' \| python3 -m json.tool` | 旧版本用 `curl` 没装,改用 `wget`(见 `docker-compose.yml` line 92 注释);改 compose 后要 `docker compose up -d --force-recreate server` 才生效 |
| `curl localhost:20080` 拒连 | `ss -tlnp \| grep 20080` | 端口冲突 → 改 compose 里 `BIND_ADDR`;容器没起 → `docker logs rsface-server \| tail -30` |
| `[persist] PG connect failed: Connection reset by peer` | `docker logs rsface-postgres \| tail -20` | PG 没 healthy 时 server 已经尝试连接;`depends_on: condition: service_healthy` 会保证顺序,如有 race 等 5s 重试 |
| `ensure_bucket failed: Connection refused`(S3) | `docker logs rsface-rustfs \| tail -10` | rustfs 没起;看 healthcheck 是否 `healthy` |
| bind mount `Permission denied` | `ls -la data/` | 旧目录是 `root:root 0755`(典型:之前用 docker volume);重命名 `data → data.old`,新建 `data/{rustfs,pg/pgdata,media}/`(当前用户拥有)。`rustfs` 容器已锁定 UID:GID = 1000:1000(见 `docker-compose.yml`),host 侧目录必须是 hyx:hyx 才能写入。 |
| 容器起不来,日志报 `port already in use` | `ss -tlnp \| grep -E '20080\|15432\|1900[01]'` | 另一个进程占了端口;在 compose 里改 HOST 端口 |
| `cargo build` 失败但 `docker compose up -d --build` 不重 build | `docker images \| grep rsface-server` | 旧镜像被缓存;`docker compose build --no-cache server` |

## 完全清理(数据会丢)

```bash
docker compose -f platform/docker-compose.yml down
rm -rf data/rustfs data/pg/pgdata data/media
```

`data/pg/rsface_dump.sqlc` **不会被删**——备份是用户资产,不与 bind mount 同生共死。

## 配置覆盖

所有 env var 在 `platform/docker-compose.yml` 有 `${VAR:-default}` 默认值;
在同目录建 `platform/.env`(可参考 `platform/.env.example`)即可覆盖。

可调参数:
- `MAX_FRAMES_VIDEO` — 单视频帧数上限(默认 3600)
- `MAX_FACE_CROPS` — 单任务人脸裁剪上限(默认 2000)
- `MIN_FACE_SIZE` — 最小人脸像素(默认 24)
- `MAX_CONCURRENT_JOBS` — 并发任务数(默认 2)
- `MAX_QUEUE_DEPTH` — 任务队列上限(默认 64)
- `JOB_TIMEOUT_SECS` / `JOB_TIMEOUT_VIDEO_SECS` — 任务超时(默认 600/3600,设 0 关闭)
- `JOB_TIMEOUT_STREAM_SECS` — 流任务超时(默认 0 = 不限,靠 cancel 控)
- `SHUTDOWN_GRACE_SECS` — SIGTERM 后任务排空宽限(compose 默认 120;二进制内置默认 180)
- `SERVER_MEM_LIMIT` — server 容器 OOM 上限(默认 2g)
- `CORS_ALLOW_ORIGIN` — 跨域白名单(留空 = 同源)

### 完整 env 参考(高级 / 部署覆盖)

下列变量在 `platform/server/src/config.rs` 都有内置默认,一般无需设置;compose 已注入
基础设施类(`BIND_ADDR` / `S3_*` / `WEB_DIR` / `TMP_DIR` / `RSFACE_CASCADE` /
`DATABASE_URL` / `LOCAL_MEDIA_DIR`),这里只列业务可调项:

| 变量 | 默认 | 说明 |
|---|---|---|
| `RSFACE_ALGO` | `haar` | 算法选择(`haar` / `cnn` / `luminance`)|
| `RSFACE_CASCADE` | `cascade.rfcf` | Haar 级联权重路径(镜像内 `/app/cascade.rfcf`)|
| `RSFACE_CNN_WEIGHTS` | 空 | 自定义 CNN 权重路径;空则用内置 starter weights |
| `RSFACE_USE_CNN` | `0` | `1` 时默认算法切到 CNN |
| `RSFACE_USE_GPU` | `1` | OpenCL squared-integral 预筛(无设备时静默回退 CPU)|
| `RSFACE_MIN_SCORE` | `0.0` | Haar 最小检测分(0.3 左右可压低 FP)|
| `RSFACE_THREAD_POOL` | `0` | 检测线程数;`0` = 物理并行度 − 1 |
| `MAX_FRAMES_VIDEO` | `3600` | 单视频处理帧数上限 |
| `MAX_FRAMES_STREAM` | `0` | 单流抽帧数上限;`0` = 不限 |
| `MAX_FACE_CROPS` | `2000` | 单任务裁剪图上限 |
| `MIN_FACE_SIZE` | `24` | 最小人脸边长 px |
| `MAX_CONCURRENT_JOBS` | `2` | 并发 job 数(1..64)|
| `MAX_QUEUE_DEPTH` | `64` | 队列背压阈值,满了 create 直接拒绝 |
| `JOB_TIMEOUT_SECS` / `JOB_TIMEOUT_VIDEO_SECS` / `JOB_TIMEOUT_STREAM_SECS` | `600` / `3600` / `0` | 各类任务硬超时秒数,`0` 关闭 |
| `SSE_KEEPALIVE_SECS` | `15` | `/events` SSE 心跳间隔 |
| `STREAM_KEEPALIVE_PERIOD` | `30` | 流 worker 无事件时的写存活周期 |
| `UPLOAD_LIMIT_IMAGE_MB` | `50` | 图片上传上限(MiB)|
| `UPLOAD_LIMIT_VIDEO_GB` | `2` | 视频上传上限(GiB)|
| `CORS_ALLOW_ORIGIN` | 空 | 跨域白名单 Origin,空 = 同源不发 CORS 头 |
| `SHUTDOWN_GRACE_SECS` | `180` | SIGTERM 排空宽限(compose 覆盖为 120)|

## GPU 部署

GPU 栈用独立 compose:
```bash
docker compose -f platform/docker-compose.gpu.yml up -d --build
```
详见 [`docker-compose.gpu.yml`](docker-compose.gpu.yml) — 需要 NVIDIA driver + nvidia-container-toolkit。
注意 GPU compose 仍用 named volumes(2026-09-18 时点),如果要从 CPU bind-mount 迁到 GPU,
需要先手动迁移数据(流程同上)。
