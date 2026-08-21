# Platform CNN Integration (2026-08-20)

把 core 已有的 CNN 人脸检测器(`src/cnn.rs`)接到 platform server,
让 Web UI / curl 用户可以一键切换 Haar / CNN 检测器。

## 改动文件

| 文件 | 变更 |
| --- | --- |
| `platform/server/src/config.rs` | 新增 `cnn_weights: Option<PathBuf>` (`RSFACE_CNN_WEIGHTS`,空串 → None) 与 `use_cnn: bool` (`RSFACE_USE_CNN=1`)。 |
| `platform/server/src/main.rs` | 启动日志多打 `mode=...` 和 `cnn: weights=...`,供运维一眼看出当前模式。 |
| `platform/server/src/jobs.rs` | 新增 `enum DetectorKind { Haar(Detector), Cnn(CnnDetector) }` 与 `build_detector(&cfg)`;`run_job` 用 `detector.detect(&frame.gray)` 统一调用,frame 循环无分支。CNN 分支把 `GrayImage` 拷成 `f32 ∈ [0,1]` 喂 `CnnDetector::detect`,再把 `CnnDetection` 映射回 core `Detection`(score ← confidence)。 |
| `platform/server/src/api.rs` | 新增 `GET /api/config`,返回当前 mode、CNN 权重路径与状态(available / missing / template)、Haar 级联路径与状态、`min_face_size`。 |
| `platform/CHANGELOG_CNN.md` | 本文件。 |

不动:`core/src/**`、`platform/web/**`、`Dockerfile`、`docker-compose.yml`、`migrations/**`、根 `Cargo.toml`。

## 新增 / 修改的 API

- `GET /api/config` (新增)

  响应:
  ```json
  {
    "mode": "haar",              // "haar" | "cnn"
    "cnn": {
      "weights_path": null,        // RSFACE_CNN_WEIGHTS 设置后的路径
      "weights_status": "n/a",     // "template" | "available" | "missing" | "n/a"
      "use_cnn": false             // RSFACE_USE_CNN
    },
    "haar": {
      "cascade_path": "cascade.rfcf",
      "cascade_status": "missing"  // "available" | "missing"
    },
    "min_face_size": 24
  }
  ```

- SSE 事件:每个 job 在 detector 就绪后多推一条 `{"type":"detector","mode":"haar|cnn"}`,
  让前端在 `EventStream` 上识别当前任务走的检测器。

## 切换方式(不改代码)

| 模式 | 启动方式 |
| --- | --- |
| Haar(默认) | 不设 `RSFACE_USE_CNN` / `RSFACE_CNN_WEIGHTS` |
| CNN + 模板权重 | `RSFACE_USE_CNN=1` |
| CNN + 自训权重 | `RSFACE_CNN_WEIGHTS=/path/to/model.cnn.bin` |

两种 CNN 模式并存:`RSFACE_CNN_WEIGHTS` 优先;为空且 `RSFACE_USE_CNN=1` 时
回落 core 内置 hand-crafted 模板权重。

`run_job` 在权重文件不存在时立即报 `cnn weights load failed (...)`,与
Haar 分支的 `cascade load failed` 行为对称。

## CPU / GPU 性能差异(参考)

CNN 与 Haar 都是纯 CPU 实现(零 OpenCL,避免重新引入 GPU 大小门控),无
GPU 加速路径。粗略数量级(从 core `bench_detect.rs` / Lena 600×600 静态
测得,具体数字以最新一次跑分为准):

| 检测器 | 600×600 单帧 | 备注 |
| --- | --- | --- |
| Haar (Detector) | 1–3 ms | 多尺度 + NMS,~3000 弱特征;GPU ≥500×500 时自动切到 OpenCL |
| CNN (CnnDetector,模板权重) | 60–150 ms | 24×24 滑动窗口 + 7 层前向;单线程;未做 image pyramid |
| CNN (CnnDetector,自训权重) | 60–150 ms | 计算量同模板权重,只差 confidence 阈值 |

**结论**:CNN 推理量是 Haar 的 ~50×(受窗口数主导),在不做 GPU 化的当前
实现下,Haar 仍是平台默认。CNN 的价值在两个场景:
1. Web UI 演示 / 对比 — 一行 env 就能让用户看到 "另一条检测管线"。
2. 集成到未来的自训模型:用 `cnn_train` 生成 `.cnn.bin`,挂到
   `RSFACE_CNN_WEIGHTS` 即生效,无需重新打包。

## 验证

- `cd platform/server && cargo build` → **0 errors, 23 warnings**(均与本
  次改动无关:`gpu` 模块的 `static_mut_refs`、`frames_since_reopen` 局部
  变量未使用,等等)。
- 三个模式本地 `cargo run` + `curl /api/config` 验证:
  - 默认 → `mode=haar`
  - `RSFACE_USE_CNN=1` → `mode=cnn, weights_status=template`
  - `RSFACE_CNN_WEIGHTS=/tmp/missing.bin` → `mode=cnn, weights_status=missing`
- 未执行 docker 部署(按约束)。
