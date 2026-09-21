# GPU 版部署验收报告 · 2026-09-21

部署对象：`platform/docker-compose.gpu.yml`（rustfs + postgres + rsface-server-gpu）

## 1. 部署结果：服务层 ✅ 通过

| 检查项 | 结果 |
|---|---|
| 三容器健康 | rsface-server-gpu / rustfs / postgres 均 healthy |
| GPU 设备注入 | `/dev/nvidia0`、`nvidiactl`、`nvidia-modeset` 存在；NVRM 540.4，CUDA 12.6 |
| 环境变量 | `RSFACE_USE_GPU=1`、cuda-backend 已编译（`--features rsface/cuda-backend`, sm_87） |
| 健康/API | `GET /api/health` = ok；Web 20080、S3 19000、PG 15432 可达 |
| 端到端检测 | 提交 biden.jpg：正确检测 1 张人脸（301×303，score 87.6），annotated/faces PNG 落 S3 |

## 2. GPU 利用：❌ 不通过（关键问题）

实测推理期间 `tegrastats` 连续采样 **GR3D_FREQ 全程 0%**——检测实际跑在 CPU。

根因（已定位到代码）：
- 平台 worker 暴露的检测器只有 **haar（默认，CPU 级联）/ cnn（CPU 手写、无预训练权重、源码注释明示 "No GPU acceleration yet"）/ luminance**。
- 真正用 `CudaBackend` 的 **SCRFD 检测器 / ArcFace 识别器**（`src/scrfd_detector.rs`、`src/arcface_recognizer.rs`，GPU 后端 `src/gpu/cuda.rs`）**没有接入平台服务**，目前只被 `src/bin` 命令行工具调用。
- 即镜像虽以 cuda-backend 编译、设备已注入，但平台 API 路径不会触发任何 GPU kernel。
- `RSFACE_USE_CNN` 未开、`RSFACE_CNN_WEIGHTS` 未配置（即便开启，cnn 也是 CPU）。

## 3. 验收结论

- **功能验收：通过**（上传→检测→产物→S3 闭环正常，稳定健康）。
- **GPU 加速验收：不通过**。当前 "GPU 版" 实为 "带 CUDA 运行时的 CPU 服务"。

## 4. 修复项（进入路线，与 persons 识别闭环合并）

1. 把 **SCRFD(GPU) 接入平台检测**，作为可选 `algo=scrfd`，提供权重配置/拉取；
2. 把 **ArcFace(GPU) embedding 接入 1:N 识别与人脸库**（与 persons 闭环同一步）；
3. 验收标准：`algo=scrfd` 推理时 **GR3D > 0%**，并给 GPU vs CPU 的 FPS/延迟对比；
4. 活体（MiniFASNet）域问题解决前不作为硬门控（见 eval-m2.md）。

> 一句话：算法层 GPU 能力是有的，缺的是"接进平台"。完成 persons + ArcFace/SCRFD 接入后复测本报告 §2。
