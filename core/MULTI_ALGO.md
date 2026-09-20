# rs-face / 3-Algo Face Detection Comparison(haar / cnn / luminance)

> **零依赖 + 统一接口** —— rs-face 的 core 层把人脸检测视为可插拔的算法族:
> 所有 detector 都实现 `rsface::face_detector::FaceDetector` trait,3 个
> zero-dep 算法对外暴露完全一致的 `detect(gray) -> Vec<Detection>` 接口,
> platform 层把它们包成一个 `DetectorKind` 枚举,跑同一个 frame 循环。
>
> 历史文档里出现过的 `yunet` / `mtcnn` / `hog` scaffold 检测器(以及
> `Maturity::Scaffold`、`src/weights/*.bin` 占位权重)已从 core 中整体删除,
> 平台不再能构造它们;历史 DB / stats 行里残留的旧算法名按原始字符串保留
> (见 §4 的兼容性说明)。

## 1. 3 个算法总览

| 算法名       | 类别        | 实现要点                                                         | 权重            | 状态 |
|--------------|-------------|------------------------------------------------------------------|-----------------|------|
| `haar`       | 经典/传统   | Viola-Jones Haar cascade,积分图 + AdaBoost 级联 + 多尺度滑窗 + NMS | cascade `.rfcf`(内置打包一份 OpenCV haarcascade,也可加载自己的) | 已有,`Maturity::Production` |
| `cnn`        | 现代 CNN    | 24×24 窗口,Conv+ReLU+Pool+FC+Sigmoid,单线程 dense 扫描 + NMS       | 可选;无权重文件时用内置 hand-crafted template | 已有,`Maturity::Experimental`(template 权重,大图慢) |
| `luminance`  | 经典/启发式 | 带状亮度 + 镜像对称 + 边缘密度多信号打分,多尺度金字塔 + NMS,无权重 | 无              | 已有,`Maturity::Experimental` |

三者全部 zero-dep、CPU-only、`Send` 但 `!Sync`(scratch 缓冲按线程独占)。

> ONNX 后端的 SCRFD 检测器仍在 core 中(`src/scrfd.rs`,feature gate 后面),
> 但 `platform/Cargo.toml` 刻意不开启任何 ONNX feature,因此 server 二进制的
> 可构造集合只有上面 3 个;`available_algos()` 也只返回这 3 个。

## 2. 统一 trait

`src/face_detector.rs`:

```rust
use rsface::image::GrayImage;
use rsface::detector::Detection;

pub trait FaceDetector: Send {
    /// 对整张灰度图做检测,返回 bounding boxes(像素坐标,按分数降序)。
    fn detect(&self, img: &GrayImage) -> Vec<Detection>;
    /// 算法名(小写、固定字符串),用于 RSFACE_ALGO / /api/config / SSE。
    fn name(&self) -> &'static str;
    /// 一句话描述,Web 端 "算法对比" 卡片底部展示(默认空串)。
    fn description(&self) -> &'static str { "" }
    /// Gray 还是 RGB;默认 Gray(所有经典检测器)。
    fn color_input(&self) -> ColorInput { ColorInput::Gray }
    /// Production / Experimental,默认 Experimental,必须显式认领 Production。
    fn maturity(&self) -> Maturity { Maturity::Experimental }
    /// 是否输出 5 点关键点(识别对齐需要),默认 false。
    fn has_landmarks(&self) -> bool { false }
    // 另有 detect_faces / detect_faces_rgb 的 blanket 实现,产出 FaceDetection。
}
```

3 个算法与 trait 的关系:
- `rsface::face_detector::HaarDetector`(newtype,包装 `src/detector.rs`
  的 `Detector`)— `name() == "haar"`,`maturity() == Production`;
- `rsface::cnn::CnnDetector`(`src/cnn/mod.rs`)— 检测入口是 inherent
  `detect(&[f32], w, h) -> Vec<CnnDetection>`(吃 [0,1] f32 缓冲,不是
  `GrayImage`),平台层做适配;
- `rsface::luminance_face::LuminanceFaceDetector`
  (`src/luminance_face.rs`)— `impl FaceDetector`,`name() == "luminance"`,
  description: "Classical luminance-pattern + symmetry detector (no weights)"。

## 3. 平台层封装

`platform/server/src/jobs.rs` 内的统一枚举(只有 3 个 variant):

```rust
pub enum DetectorKind {
    Haar(Detector),
    Cnn(CnnDetector),
    Luminance(LuminanceFaceDetector),
}

impl DetectorKind {
    pub fn detect(&self, gray: &GrayImage) -> Vec<Detection> {
        match self {
            DetectorKind::Haar(d) => d.detect(gray),
            DetectorKind::Cnn(d)   => { /* GrayImage -> f32 [0,1] 缓冲,
                                          d.detect(&f, w, h),CnnDetection -> Detection */ }
            DetectorKind::Luminance(d) => d.detect(gray),
        }
    }

    pub fn kind_name(&self) -> &'static str {
        // "haar" / "cnn" / "luminance"
    }
}
```

**选择 / 构造语义**(`build_detector(cfg, override_algo)`):

1. per-job 覆盖优先级最高:upload multipart 的 `algo` 字段(或 stream
   JSON 的 `algo`)经 `set_algo_override` 校验,只有命中
   `available_algos()` 才生效,否则静默忽略;
2. 否则看 `RSFACE_ALGO` 环境变量(`select_algo_name`):只接受
   `haar` / `cnn` / `luminance`,大小写不敏感;未知值打 warning 并回退;
3. 都没设置时走历史兼容规则:`use_cnn=true` 或 `cnn_weights` 路径存在 →
   `cnn`,否则 → `haar`。

`build_detector_by_name(name)` 是 `/compare` 端点的副入口:不依赖完整
cfg,按名字临时构造一个 detector,跑一次就 drop(`haar` 仍需通过
`Config::from_env()` 找到 cascade 文件;`cnn` 用 template 默认;
`luminance` 零外部文件,任何环境都能构造)。

```rust
pub fn available_algos() -> &'static [&'static str] {
    &["haar", "cnn", "luminance"]
}
```

## 4. 平台 API

| Endpoint | Method | 说明 |
|----------|--------|------|
| `/api/config`              | GET    | `mode` 字段 + `algo` 字段 + `available_algos: ["haar", "cnn", "luminance"]`,外加 cnn 权重 / haar cascade 文件状态 |
| `/api/jobs/{id}/compare`   | POST   | 接受 `?algos=haar,cnn,luminance&frame=N`(`frame` 可省)。对任务的第一张图(或指定 frame)串行跑多个算法,返回每个算法的 `{algo, detection_count, elapsed_ms, detections[]}`。`algos` 缺省时跑全部 3 个 |
| `/api/jobs/stats`          | GET    | 按算法字符串聚合 total/done/cancelled/error/active/avg_elapsed_ms/detections;未定算法的 active job 归入 `pending` |
| SSE 事件 `type=detector`   | —      | `algo` 字段(旧字段 `mode` 保留)+ `available_algos` 数组 |
| `RSFACE_ALGO`              | env    | 显式选择算法;接受 `haar` / `cnn` / `luminance` |
| upload `algo` 字段 / stream JSON `algo` | POST | per-job 覆盖,优先级高于 env;必须是 `available_algos()` 之一,否则忽略 |

**历史数据兼容(重要)**:`yunet` / `mtcnn` / `hog` 等已下线算法名只从
**构造路径**(`build_detector` / `build_detector_by_name` /
`set_algo_override` / `available_algos` / 前端选择列表)中移除;读取与统计
路径保持字符串透明:

- DB `jobs.algo` 列是自由 `TEXT`(migration 刻意不加 CHECK 约束),
  `persist::list_jobs` 原样读出;
- `aggregate_algo_stats` 以原始字符串为 key 分桶,旧名字照常出现在
  `/api/jobs/stats` 返回里,不 panic、不改名、不丢弃;
- `/compare` 对请求里的未知 / 旧名字先按 `available_algos()` 过滤,
  全部非法时返回 400 `no valid algos requested`,绝不尝试构造。

## 5. CLI 使用(core 二进制)

```bash
# 3 个算法走同一份 --out/--stride/... 流水线:
rs-face <INPUT> --out /tmp/result --algo haar
rs-face <INPUT> --out /tmp/result --algo cnn
rs-face <INPUT> --out /tmp/result --algo luminance

# 列出编译进来的全部算法 + maturity + 描述:
rs-face --list-algos
```

`--algo` 与 `RSFACE_ALGO` 使用同一组 tag;拼错时 CLI 会给 did-you-mean
提示并拒绝运行。

## 6. Web 端:算法对比模式

1. 平台启动后访问 `http://host:port/`,点右上角 ⚙ 按钮;
2. 弹出菜单勾选 **"Algorithm compare mode"**,状态写入 `localStorage`;
3. 上传一张图,当任务进入 preview 时,前端自动:
   - 调 `POST /api/jobs/{id}/compare?algos=haar,cnn,luminance`;
   - 3 张并排 mini canvas,每张用算法名对应的颜色画 detection 框;
   - 卡片 header 显示 `algo名 + N faces / M ms`,footer 写一句描述;
4. 关掉 toggle,panel 自动消失。

模块文件:`platform/web/compare.js`(独立 IIFE,zero-dep,样式用自带
`.rsfc-` 前缀注入,不碰 `index.html` / `style.css` 的既有规则)。
侧栏的算法过滤 chip、上传弹窗的算法下拉分别由 `index.html` 静态列表与
`app.js` 基于 `/api/config.available_algos` 动态填充,只含
haar / cnn / luminance;历史 job 上的旧算法名在预览页以大写原文展示。

## 7. 性能 / 精度

当前真实基准以仓库里的测量为准:

- Haar / CNN 的耗时与检出数见 `benches/RESULTS.md`(由
  `cargo bench --bench perf_compare -- --nocapture` 重新生成);
- CNN 在 template 权重 + 单线程 dense 24×24 扫描下,大图很慢 —— 这是
  已知限制(见 `platform/PROFILE.md` §6),不是接口问题;换真实权重 /
  SIMD / GPU dispatch 后接口不变;
- `luminance` 无权重、无乘法网络,单次 `score_window` 只做亮度/对称/
  边缘统计,定位是廉价启发式基线,精度不与 Production Haar 同级。

> 旧版文档中 dummy-weight 检测器(yunet/mtcnn/hog)的对比数字随模块一并
> 删除,不再具有参考价值;不要再引用。

## 8. 文件清单(现存相关文件)

| 文件 | 作用 |
|------|------|
| `src/face_detector.rs` | `FaceDetector` trait、`Maturity` / `ColorInput`、`HaarDetector` 适配器 |
| `src/detector.rs` + `src/haar/` | Haar 多尺度滑窗检测器;cascade 加载与内置打包 cascade(`src/haar/bundled.rs`) |
| `src/cnn/mod.rs` | `CnnDetector` / `CnnConfig` / `CnnWeights`(24×24 小 CNN) |
| `src/luminance_face.rs` | `LuminanceFaceDetector` / `LuminanceConfig`(亮度启发式,无权重) |
| `platform/server/src/jobs.rs` | `DetectorKind` 枚举(3 variant)+ `build_detector` / `build_detector_by_name` / `select_algo_name` / `available_algos` / `aggregate_algo_stats` |
| `platform/server/src/api.rs` | `/api/config`、`/api/jobs/stats`、`POST /api/jobs/{id}/compare` + `decode_to_gray` |
| `platform/server/src/config.rs` | `RSFACE_ALGO` 等环境变量解析与启动 warning |
| `platform/migrations/0001_init.sql` | `jobs.algo` 自由 TEXT 列(不加 CHECK,容忍历史算法名) |
| `platform/web/compare.js` | Web 端 "算法对比模式" 模块(3 卡片) |
| `platform/web/index.html` / `style.css` / `app.js` / `dashboard.js` | 侧栏算法 chip、上传算法下拉、stats 图表(动态基于 `available_algos`) |
| `core/MULTI_ALGO.md`(本文件) | 多算法对比设计文档 |

## 9. 验证

```bash
# core(在仓库根):
cargo fmt && cargo clippy --all-targets -- -D warnings
cargo test --lib

# platform(独立 Cargo 项目,依赖 core 的 path):
cargo fmt --manifest-path platform/Cargo.toml
cargo clippy --manifest-path platform/Cargo.toml --all-targets -- -D warnings
cargo test  --manifest-path platform/Cargo.toml --lib
```

**平台端冒烟**:

```bash
$ curl -s http://127.0.0.1:20080/api/config | jq '{algo, available_algos}'
{
  "algo": "haar",
  "available_algos": ["haar", "cnn", "luminance"]
}

$ curl -s -X POST "http://127.0.0.1:20080/api/jobs/<id>/compare?algos=haar,cnn,luminance" | jq \
    '{width, height, results: [.results[] | {algo, detection_count, elapsed_ms}]}'
{
  "width": 1126,
  "height": 661,
  "results": [
    { "algo": "haar",      "detection_count": 0, "elapsed_ms": 0 },
    { "algo": "cnn",       "detection_count": 0, "elapsed_ms": 0 },
    { "algo": "luminance", "detection_count": 0, "elapsed_ms": 0 }
  ]
}
```

(具体检出数 / 耗时随输入图片与权重变化;结构契约如上。)
