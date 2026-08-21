# rs-face 作为 SDK 使用指南

core 是一个**零运行时依赖**的 Rust crate,任何项目一行依赖即可获得
Viola-Jones(+CNN)人脸检测能力。

## 1. 依赖

```toml
[dependencies]
rsface = "0.1"            # 发布后;本地开发用 path 依赖:
# rsface = { package = "rs-face", path = "../rs-face" }
```

## 2. 最小示例:检测一张 PNG

```rust
use rsface::haar::Cascade;
use rsface::detector::{Detector, DetectorConfig};
use rsface::source::open as open_source;

fn main() -> std::io::Result<()> {
    let cascade = Cascade::load("cascade.rfcf".as_ref())?;
    let detector = Detector::new(cascade, DetectorConfig::default());

    // source::open 支持:本地文件/目录、http(s) PNG 序列、
    // mp4/rtsp(需 PATH 上有 ffmpeg)、test:// 合成流。
    let mut src = open_source("photo.png")?;
    if let Some(frame) = src.next_frame()? {
        for d in detector.detect(&frame.gray) {
            println!("face: ({}, {}, {}x{}) score={}", d.x, d.y, d.w, d.h, d.score);
        }
    }
    Ok(())
}
```

## 3. 视频/流处理 + 输出

平台(platform/)本身就是最大的 SDK 用例,核心模式:

```rust
use rsface::source::open as open_source;
use rsface::haar::Cascade;
use rsface::detector::{Detector, DetectorConfig};
use rsface::image::png::write_png_rgb;
use rsface::image::RgbImage;

let mut src = open_source("rtsp://camera/stream")?;      // ffmpeg pipe
let detector = Detector::new(Cascade::load("cascade.rfcf".as_ref())?, Default::default());

while let Some(frame) = src.next_frame()? {
    let dets = detector.detect(&frame.gray);
    if dets.is_empty() { continue; }

    // frame.rgb 可能是 None(部分源只出灰度),需要时自行转 RGB。
    let mut rgb: RgbImage = /* 拷贝或转换 */;
    for d in &dets {
        rgb.draw_rect(d.x, d.y, d.w, d.h, (0, 255, 96));
    }
    let mut png = Vec::new();
    write_png_rgb(&mut png, &rgb)?;
    // -> 存 S3 / 返回给调用方;frame.timestamp_ms 为该帧 PTS(ms)。
}
```

## 4. 常用类型速查

| 类型 | 模块 | 说明 |
|---|---|---|
| `Cascade` | `rsface::haar` | 级联分类器,`load(&Path)` 读 `.rfcf` |
| `Detector` / `DetectorConfig` | `rsface::detector` | 多尺度滑窗 + NMS;`detect(&GrayImage) -> Vec<Detection>` |
| `Detection` | `rsface::detector` | `{x, y, w, h, score}` 像素坐标 |
| `Frame` / `FrameSource` | `rsface::source` | `{index, timestamp_ms, gray, rgb}`;`open(url)` 工厂 |
| `GrayImage` / `RgbImage` | `rsface::image` | 像素缓冲 + `draw_rect`、`to_gray` 等工具 |
| `write_png_rgb/gray` | `rsface::image::png` | PNG 编码到任意 `Write` |
| `Pipeline` / `PipelineConfig` | `rsface::pipeline` | 多线程解码/检测/落盘管线(批量离线任务用) |

## 5. 约束与注意

- core **零依赖**:MJPEG 解码、HTTPS 均不在内核内,视频解码 shell 出 `ffmpeg`;
- `.rfcf` 级联由 `tools/convert_opencv_xml.py` 从 OpenCV XML 转换而来;
- CNN 检测路径(`rsface::cnn`)提供从零实现的 Conv2D+ReLU+MaxPool+FC+Sigmoid,
  可配合 `cnn_train` 训练自定义权重(见根目录 docs/algorithms.md)。
