# rsface-server 镜像:musl 静态二进制 + alpine 运行时(瘦镜像 A 方案,已应用)。
#
# 多架构:按 BuildKit 自动注入的 TARGETARCH 选择 Rust target ——
#   arm64 → aarch64-unknown-linux-musl (Jetson Orin / Apple Silicon)
#   amd64 → x86_64-unknown-linux-musl
# 两种路径都按"原生构建"假设(在目标架构的机器/VM 上构建,compose 默认
# 即为 daemon 的原生平台),因此 musl-gcc 可用;不支持跨架构交叉编译
# (amd64 主机构建 arm64 镜像需另配 cross linker,请用 `cross` 或 zigbuild)。
#
# - 编译:rust:1.97-slim + musl target,产出完全静态二进制
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

# 编译阶段固定在 build 主机的原生平台(musl-gcc 只做原生链接)。
FROM --platform=$BUILDPLATFORM rust:1.97-slim AS build
ARG TARGETARCH
WORKDIR /build

# 按 TARGETARCH 选出 musl triple,写入 /tmp/rust-target 供后续各层使用。
# 2026-09-20:旧版写死 aarch64,amd64 主机构建时 host 的 musl-gcc 是
# x86 linker,为 arm64 target 链接必然失败。
RUN case "${TARGETARCH}" in \
      arm64) echo aarch64-unknown-linux-musl ;; \
      amd64) echo x86_64-unknown-linux-musl ;; \
      *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac > /tmp/rust-target

# 预装 musl target + musl-tools(musl-gcc 只面向原生架构,与上面选取一致)。
RUN RUST_TARGET=$(cat /tmp/rust-target) \
    && rustup target add "${RUST_TARGET}" \
    && apt-get update \
    && apt-get install -y --no-install-recommends musl-tools \
    && rm -rf /var/lib/apt/lists/*

ENV CC_aarch64_unknown_linux_musl=musl-gcc \
    CC_x86_64_unknown_linux_musl=musl-gcc \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=musl-gcc \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=musl-gcc \
    RUSTFLAGS="-C target-feature=+crt-static"

# 1) 缓存层:只拷贝 manifest,先 resolve deps
COPY Cargo.toml Cargo.lock ./
COPY platform/Cargo.toml platform/Cargo.toml
RUN mkdir -p src platform/server/src \
    && echo 'pub fn _stub(){}' > src/lib.rs \
    && echo 'fn main(){}' > src/main.rs \
    && echo 'fn main(){}' > platform/server/src/main.rs \
    && mkdir -p examples tests benches src/bin \
    && for p in $(grep -A3 '^\[\[bin\]\]\|^\[\[bench\]\]' Cargo.toml \
                    | sed -n 's/^path = "\(.*\)"/\1/p'); do \
           [ -e "$p" ] || { mkdir -p "$(dirname "$p")" && : > "$p"; }; \
       done \
    && for n in $(grep -A3 '^\[\[example\]\]' Cargo.toml \
                    | sed -n 's/^name = "\(.*\)"/\1/p'); do \
           : > "examples/$n.rs"; \
       done \
    && for n in $(grep -A3 '^\[\[test\]\]' Cargo.toml \
                    | sed -n 's/^name = "\(.*\)"/\1/p'); do \
           : > "tests/$n.rs"; \
       done \
    && cargo build --release --manifest-path platform/Cargo.toml \
         --target "$(cat /tmp/rust-target)" \
    && rm -rf src platform/server/src examples tests benches

# Cargo.toml 显式声明了大量 [[bin]]/[[bench]]/[[example]]/[[test]] 目标
# (per-algorithm required-features 裁剪),cargo 加载 manifest 时就校验这些
# 路径存在 —— 缓存层只 stub 了 lib/main,缺文件会直接报
# "can't find `cascade_dump` example"。这些 target 属于 path dep,
# 本层不会编译它们(只构建 core lib + platform bin),空占位文件即可。

# 2) 真正源码
COPY src ./src
COPY platform ./platform
COPY examples ./examples
COPY benches ./benches
COPY tests ./tests
RUN touch src/lib.rs src/main.rs platform/server/src/main.rs \
    && cargo build --release --manifest-path platform/Cargo.toml \
         --target "$(cat /tmp/rust-target)" \
    # 拷到固定路径,让下游 COPY 不依赖具体 triple 名
    && cp "platform/target/$(cat /tmp/rust-target)/release/rsface-server" \
          /tmp/rsface-server

# 运行时:alpine + ffmpeg + ca-certificates
FROM alpine:3.20
RUN apk add --no-cache ffmpeg ca-certificates tini curl \
    && mkdir -p /var/rsface-media /tmp/rsface-jobs /app \
    # 所有运行时目录 + 资产用数字 UID:GID 1000:1000(hyx)持有,
    # 与 host data/* bind mount 对齐 —— docker-compose.yml 的
    # `user: "1000:1000"` 直接生效,无需镜像内 user 名匹配。
    && chown -R 1000:1000 /var/rsface-media /tmp/rsface-jobs /app

# 复制静态二进制与资产
# 二进制 group=1000:root 让 UID 1000 进程可执行(root-owned 不可写,只可执行/读)
COPY --chown=0:1000 --from=build /tmp/rsface-server /usr/local/bin/rsface-server
COPY --chown=1000:1000 platform/web /app/web
# 级联权重:仓库根的 cascade.rfcf 已移入 src/weights(随 core 嵌入式分发),
# 镜像仍需磁盘文件 —— platform server 的 haar 任务用 Cascade::load 读它
# (不是 core CLI 的嵌入式默认值)。路径必须随权重迁移同步,否则镜像构建在
# 这一步 "not found" 失败。
COPY --chown=1000:1000 src/weights/haarcascade_frontalface_default.rfcf /app/cascade.rfcf

ENV BIND_ADDR=0.0.0.0:8080 \
    S3_ENDPOINT=http://rustfs:9000 \
    S3_ACCESS_KEY=rsface \
    S3_SECRET_KEY=rsface-secret \
    S3_BUCKET=rsface \
    WEB_DIR=/app/web \
    RSFACE_CASCADE=/app/cascade.rfcf \
    TMP_DIR=/tmp/rsface-jobs \
    LOCAL_MEDIA_DIR=/var/rsface-media

# 默认以 hyx (1000:1000) 跑;compose 可覆盖但没必要。
USER 1000:1000
WORKDIR /app
EXPOSE 8080

# 容器级健康检查:HTTP /api/health(alpine 自带 curl,已安装)。
# 3 次连续失败(约 30s)才标 unhealthy,避免启动期误杀。
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -sf http://127.0.0.1:8080/api/health || exit 1

# tini:正确处理 PID 1 / 信号转发(SIGTERM → 优雅停机排空任务)
ENTRYPOINT ["/sbin/tini", "--"]
CMD ["rsface-server"]
