# rs-face Platform 端到端冒烟测试报告

> 时间: 2026-08-20  
> 环境: docker compose (rsface-server + rustfs), Debian trixie, ffmpeg 7.1.5  
> 级联: OpenCV `haarcascade_frontalface_default.xml` 转 `.rfcf`,2913 features, 25 stages

## 1. 服务可用性

| 项 | 结果 |
|---|---|
| `GET /api/health` | `{"status":"ok","service":"rsface-platform"}` |
| `GET /` (前端) | HTTP 200, 3958 bytes, 零构建 vanilla JS |
| rustfs 健康 | `0.0.0.0:9000` healthy (Docker compose 探活) |
| Server → rustfs SigV4 PUT/GET | 成功(upload 后媒体 key 在 S3 列表) |

## 2. 任务冒烟矩阵

| # | 类型 | 输入 | 结果 | 关键指标 |
|---|---|---|---|---|
| 1 | image | `biden.jpg` (ageitgey/face_recognition) | ✅ done, 0.9s | 2 张脸, score 27.67 / 8.46 |
| 2 | image | `two-people.jpg` | ✅ done | 0 张脸(角度/VJ 级联不命中,符合预期) |
| 3 | video (upload) | `face-walking.mp4` (Intel IoT DevKit) | ✅ done, 39.1s | 1830 帧, 290 帧含脸, 403 张人脸 |
| 4 | stream (URL) | `https://media.w3.org/2010/05/sintel/trailer.mp4` | ✅ running(已 cancel) | ~50fps 处理速度,76 张人脸 / 3570 帧 |

### 2.1 检测器适配性

| 测试样本 | 是否检出 | 备注 |
|---|---|---|
| biden.jpg (清晰正脸) | ✅ 2 个 bbox | 1 个高分(27.67) + 1 个低分(8.46,误检) |
| face-walking.mp4 (走路场景) | ✅ 290/1830 帧命中 | 真实场景,脸部清晰可识别 |
| two-people.jpg (合照) | ❌ 0 检出 | VJ 级联对倾斜/侧脸弱;CNN 检测器或更好 |
| Sintel trailer (动画) | 部分命中 | 动画角色偶尔触发,误检 |

## 3. 存储路径

| 路径 | 行为 |
|---|---|
| S3 PUT 成功 | 媒体写入 rustfs,key 前缀 `jobs/{id}/...` |
| S3 PUT 失败 | 自动降级本地磁盘,key 前缀 `local://` |
| `/media/{key}` | 解析 `s3://` / `local://` 前缀分发 |

> v0.1 状态:rustfs SigV4 PUT 实际可用;S3 路径在生产也可走通(本地+容器都验证过)。

## 4. 已知缺口(留给 v0.2+)

| 缺口 | 原因 | 解决方向 |
|---|---|---|
| HLS 直播流持续模式 | ffmpeg 转码 + core 多次 reopen 时序未稳 | 改用 HLS demuxer 直送 frame,不走 mp4 中转 |
| 任务历史重启即失 | JobRegistry 仅内存 | 已规划接 PostgreSQL(见任务 #8) |
| 标注闭环 | 仅有检测,无人机交互 | ROADMAP v0.3 |
| 算法自迭代 | 仅用 OpenCV Haar | 接入 cnn_train + 评测流水线(ROADMAP v0.4) |

## 5. 复现命令

```bash
# 起服务
docker compose -f platform/docker-compose.yml up -d --build

# 健康
curl -sf http://localhost:8080/api/health

# 图片
curl -F file=@platform/testdata/biden.jpg http://localhost:8080/api/jobs/image
# 视频
curl -F file=@platform/testdata/face-walking.mp4 http://localhost:8080/api/jobs/video
# 视频 URL
curl -H 'Content-Type: application/json' \
  -d '{"url":"https://media.w3.org/2010/05/sintel/trailer.mp4"}' \
  http://localhost:8080/api/jobs/stream
```

> 完整素材列表与公开 URL 见 [`testdata/INDEX.md`](INDEX.md)。
