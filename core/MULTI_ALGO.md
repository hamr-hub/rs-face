# rs-face / 5-Algo Face Detection Comparison

> **零依赖 + 统一接口** —— rs-face 的 core 层从 v0.1 起就把人脸检测视为
> 可插拔的算法族:所有 detector 都实现 `rsface::face_detector::FaceDetector`
> trait,5 个算法对外暴露完全一致的 `detect(gray) -> Vec<Detection>` 接口,
> platform 层把它们包成一个 `DetectorKind` 枚举,跑同一个 frame 循环。

## 1. 5 个算法总览

| 算法名    | 类别                  | 实现要点                                       | 权重 (bytes) | 状态                |
|-----------|-----------------------|----------------------------------------------|-------------|---------------------|
| `haar`    | 经典/传统             | Viola-Jones Haar cascade,积分图 + AdaBoost    | 0           | 已有(`cascade.rfcf` ~4KB) |
| `cnn`     | 现代 CNN              | 24×24 Conv+ReLU+Pool+FC+Sigmoid              | 0           | 已有(template)       |
| `yunet`   | 现代 anchor-based     | 5 个 anchor scale (8/16/32/64/128) + 1×1 conv + 15-dim 输出 (dx,dy,dw,dh,conf,5 landmarks) + NMS | 2048        | 新增 (`weights/yunet.bin`, dummy) |
| `mtcnn`   | 现代级联 CNN          | P-Net(12×12) → R-Net(24×24) → O-Net(48×48) 3-stage cascade + NMS | 1024×3      | 新增 (`weights/mtcnn_{pnet,rnet,onet}.bin`, dummy) |
| `hog`     | 传统 + ML             | HOG 8×8 cell / 2×2 block / 9 bin + Linear SVM 64×128 窗口 + 多尺度滑动 + NMS | 3072        | 新增 (`weights/hog_face.bin`, dummy) |

> **重要**:为了严格保持 zero-dep,yunet/mtcnn/hog 的权重都是占位 random
> bytes(`include_bytes!`),forward pass 出来后 0 检测;但 **类型、形状、
> forward、nms 全部就位**,把 fake 权重换成真权重的 `from_bytes()` 就能上线
> 真算法(例如 MTCNN 的 ~2.1MB 真实权重可以从 facenet-pytorch dump)。

## 2. 统一 trait

`src/face_detector.rs`:

```rust
use rsface::image::GrayImage;
use rsface::detector::Detection;

pub trait FaceDetector: Send {
    /// 对整张灰度图做检测,返回 bounding boxes (像素坐标)。
    fn detect(&self, img: &GrayImage) -> Vec<Detection>;
    /// 算法名(小写、固定字符串),用于 SSE / /api/config / /compare。
    fn name(&self) -> &'static str;
    /// 一句话描述,Web 端 "算法对比" 卡片底部展示。
    fn description(&self) -> &'static str { "" }
}
```

5 个算法都实现这个 trait(`impl FaceDetector for ...`):
- `Detector` (haar) — 包装 `core::Detector`
- `CnnDetector`  (cnn)  — 包装 `core::CnnDetector`
- `YunetDetector` (yunet) — 来自 `src/yunet.rs`
- `MtcnnDetector` (mtcnn) — 来自 `src/mtcnn.rs`
- `HogFaceDetector` (hog) — 来自 `src/hog_face.rs`

## 3. 平台层封装

`platform/server/src/jobs.rs` 内的统一枚举:

```rust
pub enum DetectorKind {
    Haar(Detector),
    Cnn(CnnDetector),
    Yunet(rsface::yunet::YunetDetector),
    MtCnn(rsface::mtcnn::MtcnnDetector),
    HogSvm(rsface::hog_face::HogFaceDetector),
}

impl DetectorKind {
    pub fn detect(&self, gray: &GrayImage) -> Vec<Detection> {
        match self {
            DetectorKind::Haar(d) => d.detect(gray),
            DetectorKind::Cnn(d)   => { /* f32 缓冲 + 转换 */ d.detect(&f, w, h) ... },
            DetectorKind::Yunet(d) => d.detect(gray),
            DetectorKind::MtCnn(d) => d.detect(gray),
            DetectorKind::HogSvm(d)=> d.detect(gray),
        }
    }

    pub fn kind_name(&self) -> &'static str { /* "haar" / "cnn" / "yunet" / "mtcnn" / "hog" */ }
}
```

**`build_detector` 关键代码** —— 5 个 detector 共享同一份选择逻辑:

```rust
pub fn build_detector(cfg: &Config) -> std::io::Result<DetectorKind> {
    let algo = select_algo_name(cfg);     // RSFACE_ALGO > use_cnn > haar
    match algo.as_str() {
        "haar"  => { /* load cascade, build Detector */ }
        "cnn"   => { /* load weights or template */ }
        "yunet" => Ok(DetectorKind::Yunet(YunetDetector::new(YunetConfig::default()))),
        "mtcnn" => Ok(DetectorKind::MtCnn(MtcnnDetector::new(MtcnnConfig::default()))),
        "hog"   => Ok(DetectorKind::HogSvm(HogFaceDetector::new(HogConfig::default()))),
        _ => Err(/* unsupported */),
    }
}
```

`build_detector_by_name()` 是 `/compare` 端点的副入口(不依赖 cfg,临时
按 name 构造一个 detector,跑一次就 drop)。

## 4. 平台 API

| Endpoint | Method | 说明 |
|----------|--------|------|
| `/api/config`              | GET    | 现有 `mode` 字段 + 新增 `algo` 字段 + `available_algos: ["haar", "cnn", "yunet", "mtcnn", "hog"]` |
| `/api/jobs/{id}/compare`   | POST   | 接受 `?algos=haar,cnn,yunet,mtcnn,hog&frame=N` (frame 可省,默认 0)。对任务的第一张图(或指定 frame)同时跑多个算法,返回每个算法的 `{algo, detection_count, elapsed_ms, detections[]}` |
| SSE 事件 `type=detector`   | —      | 新增 `algo` 字段 (旧字段 `mode` 保留) + `available_algos` 数组 |
| `RSFACE_ALGO`              | env    | 显式选择算法(优先级最高);接受 `haar` / `cnn` / `yunet` / `mtcnn` / `hog` |
| `--algo`                   | CLI    | 跟 `RSFACE_ALGO` 同样的 5 个选项,override 默认 haar/cnn 选择 |

## 5. CLI 使用

```bash
# 5 个算法都可以走同一份 --out/--threads/--stride/... 流水线:
rs-face <INPUT> --out /tmp/result --algo haar
rs-face <INPUT> --out /tmp/result --algo cnn
rs-face <INPUT> --out /tmp/result --algo yunet
rs-face <INPUT> --out /tmp/result --algo mtcnn
rs-face <INPUT> --out /tmp/result --algo hog
```

## 6. Web 端:算法对比模式

1. 平台启动后访问 `http://host:port/`,点右上角 ⚙ 按钮
2. 弹出菜单勾选 **"算法对比模式"**,状态写入 `localStorage`
3. 上传一张图,当任务进入 preview 时,前端自动:
   - 调 `POST /api/jobs/{id}/compare?algos=haar,cnn,yunet,mtcnn,hog`
   - 5 张并排 mini canvas,每张用算法名对应的颜色画 detection 框
   - 卡片 header 显示 `algo名 + N faces / M ms`,footer 写一句描述
4. 关掉 toggle,panel 自动消失

模块文件:`platform/web/compare.js` (独立 IIFE,zero-dep,不碰
`index.html` / `style.css`)。

## 7. 性能/精度对比(synthetic test://60, 320×240)

| 算法   | 帧数 | 检出数   | 耗时 (s) | fps      | 备注                                       |
|--------|-----|---------|---------|----------|------------------------------------------|
| haar   | 60  | 15,235  | 1.49    | 40.30    | demo cascade 过度宽松,synthetic 上很多 FP |
| cnn    | 60  | 8,100   | 608.77  | 0.10     | template weights + 单线程 dense 24×24 扫描 |
| yunet  | 60  | 60      | 6.43    | 9.34     | dummy weights,每帧 1 个 anchor 通过(5 个 scale × 1 aspect) |
| mtcnn  | 60  | 0       | 0.73    | 81.97    | dummy weights → 0 detections,但结构完整     |
| hog    | 60  | 0       | 55.32   | 1.08     | dummy SVM → 0 detections,dense 多尺度扫描 |

**真实图(平台 `/compare` 端点, two-people.jpg, 1126×661)**:

| 算法   | 检出数   | 耗时 (ms) | 备注                                       |
|--------|---------|-----------|------------------------------------------|
| haar   | 0       | 158       | OpenCV cascade 在 PGM 化后未触发           |
| cnn    | 2,347   | 80,723    | template weights + 单线程 dense 24×24     |
| yunet  | 1       | 285       | dummy weights,5 scale 各出 1 个候选         |
| mtcnn  | 0       | 0         | dummy weights 走完 3 阶段                   |
| hog    | 0       | 10,583    | dummy SVM,dense 多尺度                    |

> 这些数字反映 **当前 dummy 权重下的真实行为**。**接入真权重后,精度和
> 耗时都会有质的变化**(MTCNN 论文 0.95+ recall on FDDB @ 100 FP)。
> 接口已经就位,只需要换 weights + (可选)替换 forward 实现。

## 8. 文件清单(5 个新文件 + 5 个修改)

### 新增(5)
| 文件 | 作用 |
|------|------|
| `src/face_detector.rs` (27 行) | `FaceDetector` trait 定义 |
| `src/yunet.rs` (~210 行) | YuNet-style anchor-based detector |
| `src/mtcnn.rs` (~298 行) | MTCNN 3-stage cascade (P-Net/R-Net/O-Net) |
| `src/hog_face.rs` (~273 行) | HOG + Linear SVM,64×128 窗口 |
| `platform/web/compare.js` (~289 行) | Web 端 "算法对比模式" 模块 |
| `src/weights/yunet.bin` (2048B) | YuNet dummy 权重 |
| `src/weights/mtcnn_pnet.bin` (1024B) | MTCNN P-Net dummy 权重 |
| `src/weights/mtcnn_rnet.bin` (1024B) | MTCNN R-Net dummy 权重 |
| `src/weights/mtcnn_onet.bin` (1024B) | MTCNN O-Net dummy 权重 |
| `src/weights/hog_face.bin` (3072B) | HOG SVM dummy 权重 |
| `core/MULTI_ALGO.md` (本文件) | 5 算法对比文档 |

### 修改(4)
- `src/lib.rs` — 加 `pub mod yunet/mtcnn/hog_face/face_detector;` + re-exports
- `src/main.rs` — 加 `--algo` 参数,5 算法分发 + `run_algo_pipeline` 通用管线
- `platform/server/src/jobs.rs` — `DetectorKind` 枚举扩展 5 算法 + `build_detector` / `build_detector_by_name` / `available_algos` / `select_algo_name`
- `platform/server/src/api.rs` — `/api/config` 加 `algo` + `available_algos`,加 `POST /api/jobs/{id}/compare` 路由 + `decode_to_gray` 辅助

## 9. 验证

```bash
$ cargo build --release --workspace
   0 errors, 24 warnings (1 crates)

$ cargo test --release --workspace
   27 passed, 6 ignored (8 suites, 0.57s)
   # yunet: 3 passed, mtcnn: 3 passed, hog: 3 passed

$ cargo build --release --manifest-path platform/Cargo.toml
   0 errors, 7 warnings (2 crates)
```

**平台端**:
```bash
$ curl -s http://127.0.0.1:20080/api/config | jq
{
  "algo": "haar",
  "available_algos": ["haar", "cnn", "yunet", "mtcnn", "hog"],
  ...
}

$ curl -X POST "http://127.0.0.1:20080/api/jobs/<id>/compare?algos=haar,cnn,yunet,mtcnn,hog" | jq
{
  "job_id": "...",
  "width": 1126,
  "height": 661,
  "results": [
    { "algo": "haar",  "detection_count": 0,    "elapsed_ms": 158   },
    { "algo": "cnn",   "detection_count": 2347, "elapsed_ms": 80723 },
    { "algo": "yunet", "detection_count": 1,    "elapsed_ms": 285   },
    { "algo": "mtcnn", "detection_count": 0,    "elapsed_ms": 0     },
    { "algo": "hog",   "detection_count": 0,    "elapsed_ms": 10583 }
  ]
}
```
