# 平台审计报告 · error 任务 / M3 视频流 / M4 平台化

> 审计时间：2026-09-21（CST）。只读审计：未修改任何代码，未执行 git commit。
> 审计对象：rs-face @ /home/hyx/codespace/work/rs-face，运行栈 rsface-server-gpu / rsface-postgres / rsface-rustfs（均 healthy）。
> 基准文档：docs/PLATFORM_PRD.md（M1–M4 验收标准 + 能力清单）。

---

## 0. 2026-10-19 复核（main @ 7145660，CI 全绿后）

> 本节为对下文 §2.3 / §3.1 / §3.2 缺口清单的**事后状态复核**，基于当前 main 源码只读核对（非重新跑平台）。结论速览：**4 个 P0 中已闭环 2 个、部分闭环 1 个、仍缺 1 个**；§3.1 的 14 项里 9 项已完成或基本完成。

### 0.1 原 P0 / Top10 项现状

| 原条目（§3.2） | 现状 | 证据（当前 main） |
|---|---|---|
| **P0-1 人员 / 人脸库缺失** | 🟡 **后端已闭环，前端缺页** | migration 已到 `0008`：`platform/migrations/0005_persons.sql` 建 persons/faces 表；`gallery_handlers.rs` 提供 `GET/POST /api/persons`、`/api/persons/{id}`、`POST /api/persons/{id}/faces` 等 CRUD（api.rs:126-141 注册路由）。**但 `platform/web/` 仍无任何人员管理 UI**（grep `persons/人脸库/enroll` 无业务引用），浏览器「录入→识别」闭环在 UI 层未打通 |
| **P0-2 平台无鉴权** | 🟢 **已实现（默认关，显式开启）** | `auth_middleware`（api.rs）支持可选静态 Bearer Token：配置 `RSFACE_API_TOKEN` 后，除健康检查/Prometheus/静态资源/media 外所有 `/api/*` 需 `Authorization: Bearer`，401 + `WWW-Authenticate`，常量时间比较；SSE 用 `?access_token=` 兑底。前端 `auth.js` 401 弹窗收集并持久化。**未设置 env 时保持不鉴权（零配置本机演示），上 LAN 前必须显式开启**（README「API Token 鉴权」）。仍非多用户/RBAC，但已满足「最小门控」要求 |
| **P0-3 活体未接入平台** | 🟢 **已闭环**（精度另见 M2） | worker 已有 liveness 字段与时序门控（jobs.rs:75-105, 1276-1279 `liveness_temporal_frames`，fail-closed）；`liveness.rs` 从 `RSFACE_LIVENESS_MODELS_DIR` 加载 MiniFASNet ONNX，缺失时降级并打日志；compose 双栈都暴露 `RSFACE_LIVENESS_*`（gpu 栈默认 enabled=1）。**注意：模型精度指标 M2 仍不达标，见 docs/eval-m2.md，"接入"≠"验收通过"** |
| **P0-4 ffmpeg 失败 / 取消永久挂起** | 🟡 **取消可中断已修，早退检测待验** | 视频同步转码改 `wait_child_cancellable`（jobs.rs:1759-1777，200ms `try_wait` + cancel 即 kill）；stream 进程有 `StreamGuard` Drop kill（jobs.rs:2145）。但 §2.2 的 **stream `test://` 场景**：stream 分支对所有输入无条件 `spawn_ffmpeg_to_local`（jobs.rs:1146-1149），`is_core_native_video` 只按扩展名判定、不认 `test://`（jobs.rs:2089-2092），未见对后台 stream 子进程**早退即 fail job** 的检测，建议实测复核 |
| P1-5 RTSP 实时架构 / 15 FPS | 🔴 未见架构变更 | stream 仍是「ffmpeg 写增长 mp4 + core 反复打开」模式；无新 FPS/延迟基线产物 |
| P1-6 连续预览 / 身份叠加 | 🔴 未见 | web 仍无连续视频帧 / WebRTC |
| P1-7 OpenAPI / Swagger | 🔴 **仍缺** | platform server Cargo 无 utoipa/swagger 依赖 |
| P1-8 错误码不完整 | 🟡 **部分** | API 层已有规范码：`ssrf_blocked`(403)、`upload_too_large`、`queue_full`(429) 等（api.rs:2738-2751 有错误码契约注释）；worker/ffmpeg 层细码（ffmpeg_failed / undecodable）仍不系统 |
| P1-9 一键备份脚本 | 🔴 **仍缺** | 仓库内无 backup/restore 脚本（仅 docker-entrypoint.sh） |
| P2-10 指标 / 日志运维性 | 🟡 部分 | rate_limit、liveness 计数等有新增；FPS/延迟 histogram、JSON 日志/trace id 未见 |

新增能力（原审计之后）：SSRF 防护（blocked range + host 规范化，挡 `::ffff:` / 十进制 IP 绕过）、`file://` 拒绝、per-IP 令牌桶限流、图片输入归一化走 PPM/PGM 规避 core PNG 限制、8 个 DB migration（含 face_quality / face_verdict）。

### 0.2 建议的更新后下一步（替换原 §4）

1. ~~P0 补鉴权中间件~~ **已完成（2026-10-19）**：`RSFACE_API_TOKEN` 可选 Bearer 门控，默认关、上 LAN 前显式开启（见 README）。运营侧上线清单只剩「在部署环境里真正设置一个强 Token」；如未来需要多用户再演进到多账号/RBAC。
2. **打通人员库前端页**（后端已就绪，性价比最高）：一个「人脸库」页对接现成 `/api/persons*`，即可在浏览器走通录入→识别。
3. **实测 stream `test://` 早退**：确认 ffmpeg 早退是否会 fail job（§0.1 P0-4）；并补 FPS/延迟基线（原 §4-2 仍欠）。
4. M2 活体**精度**（独立硬骨头）：FAR≤1% / FRR≤5% 联合目标未达，EER≈5.6%，见 docs/eval-m2.md，需要更高分辨率活体网或补 attack 数据，非平台层任务。
5. 收尾：OpenAPI（utoipa）、一键备份脚本、错误码补全、孤儿任务 error 噪声治理（§3.3）。

---

## 1. jobs 表 error 任务归类（464 条）

### 1.1 总量与分布

```
status   | count
 error   |   464
 done    |   286
 cancelled|   11
```

按 error_code / kind 聚合：

| kind | error_code | 数量 | 占比 |
|---|---|---|---|
| video | orphaned | 443 | 95.5% |
| image | orphaned | 16 | 3.4% |
| stream | orphaned | 3 | 0.6% |
| video | （空） | 1 | 0.2% |
| image | （空） | 1 | 0.2% |

error 文本只有 3 种：

| error | 数量 |
|---|---|
| orphaned by server restart | 462 |
| No such file or directory (os error 2) | 1 |
| only stored blocks supported | 1 |

时间分布（created_ms 按日）：

| 日期 | error 数 | 说明 |
|---|---|---|
| 2026-08-20 | 6 | 早期手工测试（lena/two-people、RTSP 公网演示源、red.png） |
| 2026-08-21 | 25 | 早期手工/并发测试（drama-01/02.mp4 等） |
| 2026-09-18 | 26 | ep-001.mp4 短时间重复提交 26 次（队列/重启治理测试） |
| 2026-09-20 | 407 | ep-001~010.mkv 压测；11/12/14 三个整点小时段分别 128/138/136 条 |

### 1.2 归类结论

| 类别 | 数量 | 占比 | 性质判定 |
|---|---|---|---|
| A. 服务重启导致的 orphaned（running/queued 任务被启动回收器标记 error） | 462 | 99.6% | **历史测试/运维样本，非处理逻辑 bug**。462 条全部集中在 4 个开发/压测日，与压测窗口（ep-001~010.mkv 反复提交 + 容器重建）一一对应；是 startup-reaper 的预期行为（jobs.rs 中 `UPDATE ... SET status='error', error_code='orphaned'`，heartbeat 超时 30s） |
| B. 容器内文件缺失 `No such file or directory (os error 2)` | 1 | 0.2% | **真 bug（输入路径健壮性）**。job `1a0be7a58b7-0001` tiny.mkv，2026-09-20 11:01。极可能是 TMP_DIR（compose.gpu.yml 设为 `/tmp/rsface-jobs-staging`）下任务工作目录/文件被外部清理或 ffmpeg 未产出，错误未被细分码覆盖 |
| C. 非法输入解码 `only stored blocks supported` | 1 | 0.2% | **输入校验缺口（非攻击）**。job `1a01f5afb41-000c` red.png，2026-08-20。纯色 PNG 被解码器拒绝，应在上传校验阶段返回 4xx + 明确错误码，而非 error 任务 |

**与"攻击测试样本"的关系**：未发现任何一条 error 由活体攻击样本（翻拍/屏幕）产生——平台任务路径当前根本没有接入活体检测（见 §2.3）。462/464 是开发压测 + 重启回收的历史噪声；真正值得修的只有 B、C 各 1 条，且暴露的是同一类问题：**ffmpeg/文件层失败与"不可解码图片"没有在入队前被拦截，也没有独立错误码**。

佐证命令：
```bash
docker exec rsface-postgres psql -U rsface -d rsface -c \
 "SELECT kind, error_code, count(*) FROM jobs WHERE status='error' GROUP BY 1,2 ORDER BY 3 DESC"
docker exec rsface-postgres psql -U rsface -d rsface -c \
 "SELECT error, count(*) FROM jobs WHERE status='error' GROUP BY error ORDER BY 2 DESC"
```

---

## 2. M3 视频流现状与缺口

### 2.1 现状（实测）

- **RTSP 拉流**：接口存在。`POST /api/jobs/stream` 接受 `rtsp://`、`http(s)://`、`test://`（api.rs:1625-1660，有 scheme 白名单 + host 非空校验）。
- **实现方式**：不是实时帧管道，而是 ffmpeg 后台把流传码（libx264 ultrafast、**输出固定 `-r 15`**、frag mp4）写到本地文件，core 再反复打开读取增长文件（jobs.rs:798-805, 1525-1571）。架构上多了一次完整 x264 转码，延迟与吞吐双受损。
- **SSE 事件**：`GET /api/jobs/{id}/events` 存在，广播 frame/heartbeat（1s fps 心跳）/status 事件，前端用 EventSource 增量更新（app.js:1949）。
- **SSE 可视预览叠加框**：前端直播流为"双帧"对比式展示（原始 vs 标注），不是视频画面连续预览；无 WebRTC。
- **FPS/延迟基线**：**无实测基线**。当前 `/api/metrics` live_fps_max=0（服务重启后内存计数，且计数器只反映本次进程生命周期）。离线视频 done 任务耗时可佐证处理速度远低于实时：ep-001.mkv 3600 帧耗时 1009s ≈ **3.6 FPS 处理速率**（单任务，GPU 容器，haar）；其余长任务 1160–3600 帧耗时 12210–13594s（含排队等待，非纯处理时长，不可直接换算 FPS）。

### 2.2 本次只读安全测试发现（1 次，已取消）

`POST /api/jobs/stream {"url":"test://300"}` → job `1a0c4798549-0002`：
- ffmpeg 不认识 core 内部的 `test://` 合成源，进程退出成为 zombie（容器内 `ps` 见 `[ffmpeg] <defunct>`），未产出 stream.mp4；
- job 一直 `running`、frames_processed=0，SSE 端点 8s 内**零字节输出**；任务只能靠人工 cancel 结束（cancel 后状态变 done）。
- 即 **stream 路径把所有 URL 无条件喂给 ffmpeg，但 test:// 只有 core 的 source 层支持；且 ffmpeg 启动失败/早退不会让任务失败**。这是一个真 bug（P1）。

### 2.3 M3 缺口清单

| PRD M3 要求 | 状态 | 缺口 / 证据 |
|---|---|---|
| 720p RTSP 流水线 ≥15 FPS（Jetson GPU） | [ ] | 无任何 FPS 实测；离线处理速率旁证约 3.6 FPS；ffmpeg 强制二次转码到 15fps 上限 |
| 检测+活体+识别延迟 ≤300ms | [ ] | 无延迟度量；平台 worker 未接活体（容器内无 ONNX 权重，`/app` 只有 cascade.rfcf + web）；gallery 为空（data/gallery 空目录，容器无 RSFACE_GALLERY_DIR），识别也无法在流上生效 |
| 连续运行 ≥30min 无崩溃/无内存泄漏 | [ ] | 无 soak test；本次测试反证流任务在 ffmpeg 失败时永久挂起 |
| SSE/WebRTC 可视预览叠加框与身份 | ~ | SSE 事件通道有；但仅双帧对比、无连续视频预览、无身份叠加（gallery 空），无 WebRTC |
| ffmpeg 失败处理 | [ ] | spawn 后 stderr 被丢弃（/dev/null），子进程退出不被感知、不 fail job、不产生错误码（本次实测） |

---

## 3. M4 平台化差距（对照 PRD 能力清单 + M4 验收）

### 3.1 逐项核对

| # | 能力项 | 状态 | 证据 / 缺口 | 优先级 |
|---|---|---|---|---|
| 1 | Web：人脸库（人员 CRUD、一人多脸、导出） | [ ] | 无 persons 表（DB 仅 jobs/frames/faces/telemetry），前端无人员管理任何入口；gallery 靠服务器本地目录且当前为空、启动时加载一次 | **P0** |
| 2 | Web：实时预览 | [ ] | 无连续视频预览；流任务体验为双帧对比，且流任务在 ffmpeg 失败时挂死（§2.2） | P1 |
| 3 | Web：任务/系统状态、可演示 | [x] | 任务列表/过滤/统计仪表板/批量操作/时间轴齐全（index.html、dashboard.js、/api/metrics） | — |
| 4 | 浏览器走通"录入→识别" | [ ] | 录入（人员注册）完全缺失；识别依赖文件 gallery，无注册链路 | **P0** |
| 5 | OpenAPI / Swagger | [ ] | 无 utoipa/swagger 依赖与端点；API 仅在 api.rs 文件头注释中列出 | P1 |
| 6 | Token 鉴权 | [ ] | 全无鉴权中间件；grep 仅见 S3 签名与 telemetry 字段过滤中的 "bearer" 字符串；局域网内任何人可调全部接口（含删任务） | **P0** |
| 7 | 统一错误码 | ~ | 有 error_response(error_code) 结构与 jobs.error_code 列，但码表未文档化、码不全（文件缺失/解码失败/ffmpeg 失败都落到空码或裸文本）；无 OpenAPI 对应 | P1 |
| 8 | 健康检查 | [x] | `/api/health` 浅探 + `/api/health/deep`（PG+S3）实测均 ok；compose healthcheck 使用 | — |
| 9 | Prometheus 指标 | [x] | `/metrics` 暴露 12+ rsface_* gauge（jobs/frames/detections/avg_job_ms），实测可抓取。缺口：无 FPS/延迟/活体 histogram，指标为进程内存口径（重启清零） | P2 |
| 10 | 结构化日志 | ~ | tracing + EnvFilter，但默认是 fmt 纯文本行（main.rs init_tracing），非 JSON 结构化；无请求 trace id 中间件 | P2 |
| 11 | 备份恢复 | ~ | DOCKER.md 有手工 pg_dump/pg_restore 命令（自定义格式），但无一键脚本；rustfs/media 备份无文档；"一键备份/恢复文档"未达标 | P1 |
| 12 | 空机冷部署 ≤30min | ~ | MINIMUM_CONFIG.md + DOCKER.md 存在，compose 自动跑 4 个 SQL migration；但空 data/pg 时 initdb 后需手工 pg_restore 的说明与"全新空库直接可用"矛盾，且未做过限时实测；gallery/权重初始化无引导 | P2 |
| 13 | GPU/CPU 双部署 | [x] | docker-compose.yml + docker-compose.gpu.yml 均在，gpu 容器当前 healthy | — |
| 14 | 活体门控接入平台 | [ ] | core 有 liveness.rs/liveness_detector.rs，但平台 worker 未调用、容器无 MiniFASNet ONNX；属 M2 遗留，直接阻塞 M3"检测+活体" | **P0** |

### 3.2 关键缺口 Top 10（跨 M3/M4，按优先级）

1. **P0 人员/人脸库完全缺失**：无 persons 数据模型、无 CRUD API、无前端页面，"录入→识别"闭环不存在。
2. **P0 平台无任何鉴权**：全部 API 裸奔于绑定 0.0.0.0:20080 的端口。
3. **P0 活体未接入平台任务**：无 ONNX 权重、worker 无调用，M2/M3 验收前提缺失。
4. **P0 流任务 ffmpeg 失败即永久挂起**：test://300 实测，0 帧、SSE 无输出、无错误码（api 允许的 test:// 在 stream 路径不可用）。
5. **P1 RTSP 实时架构不达标**：ffmpeg 二次转码 + 增长文件读取，无 FPS/延迟基线；旁证处理速率约 3.6 FPS，距 15 FPS 差距大。
6. **P1 无连续可视预览/身份叠加**：仅双帧对比，无 SSE 视频帧或 WebRTC。
7. **P1 无 OpenAPI/Swagger**：接口无机器可读文档，SDK 无法生成。
8. **P1 错误码不完整、未文档化**：tiny.mkv / red.png / ffmpeg 失败均无规范码。
9. **P1 备份恢复无一键脚本、媒体/rustfs 备份缺失**。
10. **P2 指标与日志运维性不足**：指标进程内存口径、无 FPS/延迟 histogram；日志非 JSON、无 trace id；冷部署 30min 未实测。

### 3.3 error 噪声本身的运维缺口（补充）

462 条 orphaned 虽非 bug，但占 error 总量 99.6%，会淹没真实失败信号。建议（仅建议，不在本次执行）：
- 重启回收的 running 任务标为 `interrupted`/自动 requeue（视频/图片任务输入可重新获得时可安全重试），与真正处理失败分离；
- 提供 error 列表按 error_code 的默认过滤与归档策略。

---

## 4. 建议的最小下一步（按顺序，最小成本解锁后续验收）

1. **修流任务存活检测（半天）**：捕获 ffmpeg 子进程退出（wait 任务）、stderr 落日志，退出即 `set_error_code('ffmpeg_failed')` 并 fail job；stream 路径对 `test://` 直走 core source 不经 ffmpeg。修复后用 test://300 重测，应见 SSE heartbeat fps>0。
2. **补一张 FPS/延迟基线（1 小时）**：用 test:// 合成源 + 一个真实 720p mp4 循环，记录每帧处理耗时与 fps（先不接活体），作为 M3 优化前基线。
3. **加最小鉴权（半天）**：单 token 中间件（env 配置），保护 /api/*，前端登录/存储 token；解锁"可上 LAN 演示"。
4. **建 persons/gallery 数据模型 + 最小 CRUD（2–3 天）**：一张 persons 表 + faces 关联 + 4 个端点 + 前端一个"人脸库"页，打通浏览器"录入→识别"。
5. 之后再排：活体 ONNX 进 GPU 镜像并接入 worker → OpenAPI（utoipa）→ 一键备份脚本。

---

## 附：审计依据命令索引

- `docker exec rsface-postgres psql -U rsface -d rsface -c "\d jobs"` / 各聚合 SELECT（§1）
- `curl -s http://localhost:20080/api/health{,/deep}` / `/api/config` / `/api/metrics` / `/metrics`（§3）
- `curl -X POST .../api/jobs/stream -d '{"url":"test://300"}'` + `/api/jobs/{id}` + `/events` + `docker exec ... ps` / `find`（§2.2）
- 代码：platform/server/src/api.rs（路由、start_stream、错误响应）、jobs.rs（heartbeat、stream ffmpeg、spawn_ffmpeg_to_local）、main.rs（tracing、migration）、metrics.rs；platform/web/index.html、app.js
- 文档：platform/DOCKER.md（备份恢复）、MINIMUM_CONFIG.md（冷部署）、platform/docs/ROADMAP.md
