# Platform Performance & Telemetry Pass (2026-09-18)

按 `/goal`:前端响应速度优化 + 用户行为埋点 + 服务端 API 响应优化 + 服务端核心功能加固。

## 一、后端 API 响应优化

| 改动 | 收益 |
| --- | --- |
| `tower-http::CompressionLayer` 全局 gzip | JSON 响应 70-90% 体积压缩(已验证 `/api/config` 走 gzip) |
| `tower-http::SetResponseHeaderLayer` 注入 `Cache-Control: public, max-age=600, stale-while-revalidate=86400` | 前端静态资源首次加载后,reload 走 304 |
| `serve_static_with_304()` 处理 `If-None-Match` | 浏览器 reload `/app.js` / `/style.css` 0 字节 body 命中(已验证) |
| 新模块 `cache::TtlCache` 零依赖 TTL 缓存 | 给 `/api/config` 和 `/api/metrics` 加可配置 TTL |
| `/api/config` 1h TTL 缓存(启动期 immutable) | 多 tab 同时打开 / 重复请求 0 次 cfg 读 |
| `/api/metrics` 1s TTL dedup | 2s 轮询 50% 命中,1s 内多 tab 共用 snapshot |
| 静态文件 ETag = `W/"rsface-{len:x}-{mtime:x}"` | 弱 ETag,基于文件 mtime,reload 0 字节 body |

## 二、前端响应速度优化

| 改动 | 收益 |
| --- | --- |
| `<script defer>` 所有模块(`toast / modal / dashboard / visibility / telemetry / app`) | 解析期并行下载,DOMContentLoaded 前执行,不阻塞首屏渲染 |
| `kpi.refresh()` 用 `AbortController` 取消旧请求 | 慢响应不再让 KPI 请求堆积 |
| `renderFaceGrid()` 用 `DocumentFragment.replaceChildren()` 一次性插入 | 100 张脸卡:100 次 layout → 1 次 |
| search debounce 120ms → 80ms | 输入响应更跟手 |
| `grid.querySelectorAll('img[data-src]')` 替换为局部 `card.querySelector(...)` | O(N) 全 grid 扫描 → 单 card 局部查找 |

## 三、用户行为埋点(端到端)

| 改动 | 收益 |
| --- | --- |
| `POST /api/telemetry` 端点(批量 ingest) | 一次性批量上报,降低 HTTP 开销 |
| 前端 `telemetry.js` 模块 | 自动捕获 5 类事件:`page_view` / `api_call` / `api_error` / `js_error` / `js_unhandled` |
| `window.fetch` wrap + 拦截 | 自动给所有 `/api/*` 请求打点(endpoint + status + duration_ms),不抓 body/headers |
| `sendBeacon` 兜底 `visibilitychange` / `pagehide` / `beforeunload` | 页面卸载不丢点 |
| 业务事件 `__track('upload_submitted' / 'upload_failed' / 'upload_exception', { kind, size_kb, algo, round_trip_ms })` | 业务漏斗分析(上传路径耗时 / 失败率 / 算法偏好) |
| 服务端二次过滤(PII: `s3://` / `local://` / `inline://` / `/media/` / `Authorization`) | 客户端绕过 / 异常数据也不会进 DB |
| 表 `telemetry(id BIGSERIAL, name TEXT, ts_ms BIGINT, received_ms BIGINT, session TEXT, path TEXT, props JSONB)` | 极简 schema,运维按月归档 |

## 四、服务端核心优化

| 改动 | 收益 |
| --- | --- |
| `state._kpiAbort` 单例 AbortController | KPI 取消并发 |
| 路径 `cache.rs` 单元测试 4 条 | TTL 过期 / 并发安全 / put 覆盖 |

## 验证

- `cargo test --release`:31 passed (含 cache 模块新单测)
- `cargo build --release`:0 error, 0 warning
- 端到端 curl:
  - `GET /api/config` → 200 + `content-encoding: gzip` + `cache-control: ...`
  - `GET /style.css` → 200 + `etag: W/"rsface-..."` + `cache-control: ...`
  - `GET /style.css` + `If-None-Match: W/"..."` → 304 (0 字节 body)
  - `POST /api/telemetry` → `{"accepted":3,"ok":true}`
  - `POST /api/telemetry` 含 `s3://` → `{"accepted":0,"ok":true}` (服务端二次过滤命中)
  - `POST /api/jobs/image` + `algo=luminance` → job 创建,检测跑通,1s 内完成
  - `POST /api/jobs/{id}/compare` → 5 种算法对比,cnn 检出 1143 个,haar 报 cascade 缺失错误

## 改动文件清单

| 文件 | 变更 |
| --- | --- |
| `Cargo.toml` | 新增 `tower-http = "0.6"` (compression-gzip + set-header) |
| `server/src/cache.rs` | 新文件:`TtlCache` 实现 + 4 条单测 |
| `server/src/main.rs` | 注册 `cache` 模块 + `ResponseCaches` 实例化 |
| `server/src/api.rs` | router 接受 `ResponseCaches` + `CompressionLayer` + `SetResponseHeaderLayer`;所有 handler 改用 `(state, caches)` 元组 state;新增 `serve_static_with_304()` / `etag_eq()` / `telemetry_ingest()` / `TelemetryBatch` / `is_safe_event()`;`metrics` / `config_info` 走 TTL 缓存 |
| `server/src/persist.rs` | 新增 `TelemetryEvent` 结构 + `insert_telemetry_batch()` UNNEST 批量写入;`migrate()` 支持多迁移文件 |
| `migrations/0002_telemetry.sql` | 新文件:`telemetry` 表 |
| `web/index.html` | `<script>` 加 `defer`,新增 `<script defer src="/telemetry.js">` |
| `web/telemetry.js` | 新文件:自动采集 + sendBeacon 兜底 + PII 过滤 + 业务 API(`__track`) |
| `web/app.js` | KPI `AbortController`;`renderFaceGrid` DocumentFragment + 单 card 局部查找;search debounce 80ms;upload/stream 三类业务事件 `__track()` 埋点 |

## 注意事项

- 内存模式(`DATABASE_URL=""` 空)下埋点只走 stdout,不上表 — 启动日志可观察 `[telemetry] batch N events, accepted=M first.name=...`
- 有 DB 时异步批量写;失败仅 eprintln,不阻塞 ingest 响应
- 客户端禁用埋点:localStorage `rsface.telemetry.muted = '1'`
- 服务端兜底过滤是防御性,客户端已扫过,这里再扫一次防绕过