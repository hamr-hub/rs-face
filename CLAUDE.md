# rs-face 项目规则

> **目录结构 / 新增文件位置 / "Do NOT" 硬约束 → 先读 [`STRUCTURE.md`](STRUCTURE.md)。**
> 本文件只列**操作层**规则（怎么部署、怎么开发、怎么集成测试）。

## 部署 / 启动 / 集成测试 → Docker 是唯一标准方式

platform 服务(rustfs + postgres + rsface-server)**必须**通过 `docker compose` 启动,
**不允许在 host 上裸跑 `cargo run` 跑 server**。所有 e2e 验证、备份、恢复都走容器。

- **权威文档**:`platform/DOCKER.md`(部署 / 启动 / 测试 / 备份 / 故障排查)
- **Compose 文件**:`platform/docker-compose.yml`(CPU,host 网络,bind mount)
- **常用命令**:`docker compose -f platform/docker-compose.yml up -d --build` / `down` / `logs -f --tail=100` / `ps`;`bash platform/scripts/docker-smoke.sh` 跑 e2e
- **数据位置**(bind mount,跟随代码,可见可备份):`data/{rustfs,pg/pgdata,media}/`

## 前端开发(`platform/web/`)→ pnpm dev 直连 docker 后端

前端是**零依赖 vanilla JS**(5181 行,无 build tool)。开发流程:

```bash
docker compose -f platform/docker-compose.yml up -d --build  # 后端先起,listen 0.0.0.0:20080
pnpm install          # 一次性:只装 vite 作 dev server
pnpm dev              # vite :5173,proxy /api → docker :20080,改代码立即 HMR
```

- **入口**:`package.json` + `vite.config.js` 在仓库根(`root: 'platform/web'`)
- **proxy**: `/api/*` 和 `/events` → `http://localhost:20080`(`RSFACE_BACKEND` env 可改)
- **改前端代码** → vite HMR 立刻在 :5173 看到,**不要碰 docker 容器里的 `/app/web/`**——会被下次 `docker compose up -d --build` 覆盖
- **生产构建**: vite build 写 `web-dist/`,但**生产镜像仍走 `platform/web/` 直拷到容器**(见 `platform/Dockerfile`),不需要 build step

## 算法核心 (`src/`) → cargo 直接跑,不需要 Docker

`cargo test`、`cargo clippy`、`cargo bench` 等都走原生 cargo
路径,这是开发期的 fast-iter 工具。CI 也用 cargo(ubuntu + macOS matrix,见
`.github/workflows/ci.yml`)。

**不要**把 `cargo build/test/clippy` 包装到 docker 里——会让本地迭代变慢 5-10 倍。
Docker 的核心价值是统一 platform 服务的运行时环境(ffmpeg + postgres + rustfs +
静态二进制 + non-root user),不是把所有 cargo 命令都套层壳。

## 数据迁移提醒

- `data/` 目录在仓库根,**显式 bind mount**,不用 docker named volume。
- 首次 `docker compose -f platform/docker-compose.yml up -d --build` 时,
  如果 `data/pg/pgdata/` 是空目录,postgres 容器会自动 initdb 一个空库——
  没有任何表,需用 `docker exec -i rsface-postgres pg_restore ...` 还原 dump。
- 旧版本(named volume: `platform_rsface-media` / `platform_pg-data` /
  `platform_rustfs-data`)在 2026-09-18 已退役;新部署只用 `data/` bind mount。

## 文档索引

- 项目总览:[`README.md`](README.md)
- 贡献流程:[`CONTRIBUTING.md`](CONTRIBUTING.md)
- Docker 权威指南:[`platform/DOCKER.md`](platform/DOCKER.md)
- 算法/架构/性能:[`docs/INDEX.md`](docs/INDEX.md)
- 平台设计:[`platform/docs/PLATFORM_DESIGN.md`](platform/docs/PLATFORM_DESIGN.md)
