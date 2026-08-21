# rsface-server 镜像:musl 静态二进制 + alpine 运行时(瘦镜像 A 方案,已应用)。
#
# 目标架构:arm64 (Jetson Orin 部署环境)。
# - 编译:rust:1.97-slim + aarch64-unknown-linux-musl target,产出完全静态二进制
# - 运行时:alpine:3.20 + ffmpeg + ca-certificates + tini
# - 镜像大小:~125 MB(相比原 521 MB 减少 76%)
#
# 历史方案(仅保留作参考,不推荐使用):
# - Dockerfile.alternative-strip:strip + lto 但保留 debian-slim + apt ffmpeg(只省 ~7 MB)
# - Dockerfile.alternative-static-ffmpeg:静态 ffmpeg,jrottenberg/ffmpeg 仅 amd64,与本机不兼容
#
# 构建:
#   docker build -f platform/Dockerfile -t rsface-server:slim .
# 或走 compose:
#   docker compose -f platform/docker-compose.yml up -d --build server

FROM rust:1.97-slim AS build
WORKDIR /build

# 预装 musl target
RUN rustup target add aarch64-unknown-linux-musl \
    && apt-get update \
    && apt-get install -y --no-install-recommends musl-tools \
    && rm -rf /var/lib/apt/lists/*

ENV CC_aarch64_unknown_linux_musl=musl-gcc \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=musl-gcc \
    RUSTFLAGS="-C target-feature=+crt-static"

# 1) 缓存层:只拷贝 manifest,先 resolve deps
COPY Cargo.toml Cargo.lock ./
COPY platform/Cargo.toml platform/Cargo.toml
RUN mkdir -p src platform/server/src \
    && echo 'pub fn _stub(){}' > src/lib.rs \
    && echo 'fn main(){}' > src/main.rs \
    && echo 'fn main(){}' > platform/server/src/main.rs \
    && cargo build --release --manifest-path platform/Cargo.toml --target aarch64-unknown-linux-musl \
    && rm -rf src platform/server/src

# 2) 真正源码
COPY src ./src
COPY platform ./platform
RUN touch src/lib.rs src/main.rs platform/server/src/main.rs \
    && cargo build --release --manifest-path platform/Cargo.toml --target aarch64-unknown-linux-musl

# 运行时:alpine + ffmpeg + ca-certificates
FROM alpine:3.20
RUN apk add --no-cache ffmpeg ca-certificates tini curl \
    && mkdir -p /var/rsface-media /tmp/rsface-jobs /app \
    # 非 root 运行:rsface 用户(UID 10001),媒体/临时目录授权
    && addgroup -S -g 10001 rsface \
    && adduser -S -D -u 10001 -G rsface rsface \
    && chown -R rsface:rsface /var/rsface-media /tmp/rsface-jobs /app

# 复制静态二进制与资产
COPY --from=build /build/platform/target/aarch64-unknown-linux-musl/release/rsface-server /usr/local/bin/rsface-server
COPY --chown=rsface:rsface platform/web /app/web
COPY cascade.rfcf /app/cascade.rfcf

ENV BIND_ADDR=0.0.0.0:8080 \
    S3_ENDPOINT=http://rustfs:9000 \
    S3_ACCESS_KEY=rsface \
    S3_SECRET_KEY=rsface-secret \
    S3_BUCKET=rsface \
    WEB_DIR=/app/web \
    RSFACE_CASCADE=/app/cascade.rfcf \
    TMP_DIR=/tmp/rsface-jobs \
    LOCAL_MEDIA_DIR=/var/rsface-media

USER rsface
WORKDIR /app
EXPOSE 8080

# 容器级健康检查:HTTP /api/health(alpine 自带 curl,已安装)。
# 3 次连续失败(约 30s)才标 unhealthy,避免启动期误杀。
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -sf http://127.0.0.1:8080/api/health || exit 1

# tini:正确处理 PID 1 / 信号转发(SIGTERM → 优雅停机排空任务)
ENTRYPOINT ["/sbin/tini", "--"]
CMD ["rsface-server"]
