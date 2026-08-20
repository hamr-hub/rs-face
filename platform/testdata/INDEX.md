# rs-face Platform 测试素材索引

> 用 curl / 平台 UI 复现人脸识别流水时,把下面这些 URL / 本地路径直接喂给
> `/api/jobs/image`、`/api/jobs/video`、`/api/jobs/stream` 即可。
> 一切素材均**公开可下载**或为本地自生成,本目录已下载几份副本供离线复现。

## 0. 本地副本(已下载,平台 docker 挂载即用)

```
platform/testdata/
├── bbb-360-10s.mp4       991 KB    Big Buck Bunny 360p 10s,通用视频(无脸,仅验流水线)
├── face-walking.mp4      6.4 MB    Intel IoT DevKit,单人走过镜头 61s,含清晰人脸
├── face-pose-male.mp4    15.5 MB   Intel IoT DevKit,男性多角度头部 30s+
├── lena.jpg              标准测试图(无脸,验证图片通道)
├── biden.jpg             ageitgey/face_recognition 例图,单人正脸
└── two-people.jpg        ageitgey/face_recognition 例图,双人正脸
```

直接通过 multipart 上传(文件路径 → curl `-F file=@…` 或 Web UI 拖拽)。

## 1. 公开视频 URL(直连可用)

| 名称 | 用途 | URL |
|---|---|---|
| Sintel trailer | 通用 52s 1080p,验证大文件 | <https://media.w3.org/2010/05/sintel/trailer.mp4> |
| Big Buck Bunny 10s/1MB | 360p 短片,验证小文件快速跑通 | <https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4> |
| face-demographics-walking | Intel 公开样片,含人 | <https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4> |
| head-pose-face-detection-male | Intel 公开样片,人脸 | <https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/head-pose-face-detection-male.mp4> |
| head-pose-face-detection-female | Intel 公开样片,人脸 | <https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/head-pose-face-detection-female.mp4> |
| sample-5s | samplelib 5s 短片 | <https://download.samplelib.com/mp4/sample-5s.mp4> |

> 平台 `/api/jobs/video` 接受 multipart 上传;`/api/jobs/stream` 接受 URL 时
> core 会用 ffmpeg pipe 拉流,这意味着 URL 给视频也走"流式"路径,行为一致。

## 2. 公开 HLS 直播流(已验证可达)

| 来源 | 用途 | URL |
|---|---|---|
| Mux x36xhzz | 24h 循环测试流,纯内容无脸 | <https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8> |
| Mux test_001 | 24h 循环测试流,纯内容无脸 | <https://test-streams.mux.dev/test_001/stream.m3u8> |
| Apple BipBop | Apple 官方 HLS 示例,重复片段 | <https://devstreaming-cdn.apple.com/videos/streaming/examples/img_bipbop_adv_example_ts/master.m3u8> |
| Unified Streaming Tears of Steel | 30s 广告片 HLS | <https://demo.unified-streaming.com/k8s/features/stable/video/tears-of-steel/tears-of-steel.ism/.m3u8> |

> core 会把 HLS 当 ffmpeg pipe 拉流,逐帧检测;`/api/jobs/stream` 收到 HLS
> 会一直跑,前端"停止"按钮通过 cancel 标志位让 core 退出循环。

## 3. RTSP 公开地址(本环境实测均不可达,文档化)

| 来源 | 备注 |
|---|---|
| `rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov` | 防火/超时,多数公司网段不可达 |
| `rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov` | 同上 |

### 本地 RTSP / HLS 自建(本目录已附脚本)

为不依赖外网,提供 `scripts/serve_hls.sh` —— 用 ffmpeg 把 `face-walking.mp4` 循环切片为
本地 HLS,平台把它当 `http://<host>:18080/stream.m3u8` 拉流。便于内网 demo。

## 4. 公开图片(单图)

| 名称 | URL |
|---|---|
| OpenCV 标准 Lena | <https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg> |
| dlib 样例脸 1 | <https://raw.githubusercontent.com/davisking/dlib/master/examples/faces/2007_007763.jpg> |
| dlib 样例脸 2 | <https://raw.githubusercontent.com/davisking/dlib/master/examples/faces/2009_004587.jpg> |
| biden | <https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg> |
| two_people | <https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/two_people.jpg> |

## 5. 平台内置合成流(无需任何外部素材)

```
test://120         # 120 帧合成图(中心亮边缘暗),demo 级联能命中,verify 流水线
```

## 6. 复现脚本(curl)

```bash
# 图片
curl -F file=@platform/testdata/biden.jpg http://localhost:8080/api/jobs/image

# 视频
curl -F file=@platform/testdata/face-walking.mp4 http://localhost:8080/api/jobs/video

# 视频 URL
curl -H 'Content-Type: application/json' \
  -d '{"url":"https://media.w3.org/2010/05/sintel/trailer.mp4"}' \
  http://localhost:8080/api/jobs/stream

# HLS 直播
curl -H 'Content-Type: application/json' \
  -d '{"url":"https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"}' \
  http://localhost:8080/api/jobs/stream
```

返回 `{job_id}` 后访问 `http://localhost:8080/?id=<job_id>` 或
`GET /api/jobs/<id>` 看结果;`/api/jobs/<id>/events` 是 SSE。
