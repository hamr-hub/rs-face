# 远端分支盘点 · 2026-10-19

> 盘点基准：main @ `7145660`（CI 修复 PR #14 合并后，20 checks 全绿）。
> 方法：`git fetch --prune` 后，对每个远端分支统计 ahead/behind，结合 main 的 merge 提交记录与源码 grep 核对「独有成果是否已进 main」。
> 注意：main 上大量合并是 **squash / 自定义 `merge:` 提交**，不保留原始 commit 拓扑，因此 `git branch --merged` 会漏报，不能仅凭 commit 历史判断，必须看代码内容。

---

## 1. 已可安全删除（9 个）— main 有明确 merge 登记，且关键成果 grep 均已在 main

这些分支 main 的 `git log` 中有对应的 `Merge branch '...'` / `merge: ...` 提交记录，且抽查其标志性文件/功能均已存在于 main。

| 分支 | main 中的合并登记 |
|---|---|
| `codex/minifas-temporal` | `Merge branch 'codex/minifas-temporal': UI panel placement, ... temporal liveness threshold` |
| `perf/variance-prefilter-simd` | `merge: SIMD-batched variance pre-filter (...)` |
| `feat/web-compare-timeline-batch` | `Merge branch 'feat/web-compare-timeline-batch'` |
| `platform/robustness-2` | `Merge branch 'platform/robustness-2'` |
| `feat/landmarks-and-tracker` | `Merge branch 'feat/landmarks-and-tracker'` |
| `accuracy/cascade-nms-golden` | `Merge branch 'accuracy/cascade-nms-golden'` |
| `perf/simd-hotpath` | `Merge branch 'perf/simd-hotpath'` |
| `perf/diagonal-haar-and-jpeg` | `Merge branch 'perf/diagonal-haar-and-jpeg'` |
| `feat/swiss-army-knife` | `Merge pull request #3 ... (feat/swiss-army-knife)` |

> 删除前建议再跑一次 `git diff main origin/<branch> -- <该分支标志性文件>` 抽查；删除命令：
> `git push origin --delete <branch>`（远端分支删除后仍可在 GitHub 上按 SHA 恢复一段时间）。

---

## 2. 内容基本已进 main、但无干净合并拓扑（6 个）— 建议删除，删除前人工抽查

这些分支 ahead 21~68、behind 145+，其标志性成果在当前 main 已能找到对应实现，应是被后续 sprint / squash 吸收，但 main 无一一对应的 merge 记录。

| 分支 | 标志成果 | 在 main 的核对结果 |
|---|---|---|
| `feat/new-algorithms` | skin / LBP / HOG 检测器 | ✅ `src/skin_face.rs` `src/lbp_face.rs` `src/hog_face.rs` 均在 |
| `chore/rustdoc-gate` | 无警告 rustdoc + CI doc gate | ✅ CI 有 3 处 `cargo doc` 门禁 |
| `accuracy/ensemble-and-calibration-v2` | ensemble / 标定 | ✅ ensemble/calibration 概念已散布在 haar/lbph/batch 等 |
| `chore/finish-broad-sweep` | streaming zip body、tracing subscriber、集成测试 | 🟡 zip 有流式 `finish_into`；main.rs 有 tracing_subscriber；对应 PR #12 已 merge |
| `perf/speedup-sprint-2` | cascade 预解析/预算阈值、cuda feature、Dockerfile.gpu | 🟡 Cargo.toml 有 cuda feature；cascade 已多轮优化（建议抽查） |
| `test/ci-hardening-2` | 4 个新测试套件 | 🔴 **未进 main**：`tests/{cli_flags,property_face,video_chunking,cross_algo}.rs` 均不存在（见 §3） |

> 前 5 个基本可删；**`test/ci-hardening-2` 不建议直接删**——它带的 4 个测试套件 main 没有，属可打捞的增量。

---

## 3. 含未回收增量（2 个）— 暂不删，建议打捞后再删

| 分支 | 未进 main 的增量 | 处置建议 |
|---|---|---|
| `test/ci-hardening-2` (ahead 58) | `tests/property_face.rs`(618)、`video_chunking.rs`(363)、`cli_flags.rs`、cross_algo 等约 47 个测试文件；但同时基于很老的代码（其 diff 里 main 现在的 jpeg/liveness/golden 测试在它看来是"删除"） | 不要整支合并。按文件**手工 cherry-pick / 移植**仍有价值的测试（property_face、video_chunking 最可能独立有用），移植到当前 main 并跑绿后再删分支 |
| `platform/stability-hardening-2` (ahead 34) | 平台 `error_codes.rs` / `slow_log.rs` / `retry.rs`（PG/S3 指数退避）模块 main 缺失；但 rate_limit 已进 main | 评估 retry（PG/S3 退避）和统一错误码是否仍需要（与 audit §3.1 第 7/8 项呼应）；需要则重做为基于当前 main 的新分支，旧分支删除 |

---

## 4. 待核（1 个）

| 分支 | 情况 |
|---|---|
| `feat/web-ux-completeness-2` (ahead 67) | 含「running stats dashboard + per-algo breakdown + slowest jobs」前端，当前 `platform/web/` grep 不到 per-algo/slowest 仪表板，疑似**未回收**；另有 `Db::set_archived` 等后端改动。删除前需逐文件 diff 确认该前端特性是否已被别的 UI 工作取代 |

---

## 5. 一句话结论

- **9 个已合并分支可立即删**（§1）；
- **5 个内容已吸收的可删**（§2 除 test/ci-hardening-2）；
- **2 个有未回收代码（test 套件、平台 retry/错误码）+ 1 个待核前端**，先打捞/核对再删（§3、§4）。
- 全部 17 个未合并分支都 behind main 145+ 提交，**没有任何一支适合直接 merge/rebase**，只能作为「打捞个别文件」的来源。
