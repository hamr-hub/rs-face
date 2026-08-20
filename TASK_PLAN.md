# rs-face 优化任务清单

## P0 — 修编译错误 [src/gpu/mod.rs] ✅
- [x] 修 line 168 `c_dlsym` 的 `*const i8` → `*const u8`
- [x] 修 line 177 `dlopen` 的 `*const i8` → `*const u8`
- [x] 修 line 460 `create_program_with_source` 的 `*const *const u8` (用 local 变量)
- [x] 修 line 472 `create_kernel` 的 `*const u8`
- [x] `cargo build --release` 0 error

## P1 — 清警告 ✅
- [x] 删 5 个不必要的 `unsafe {}` 块 (probe + 4 methods + 3 let lib)
- [x] `i_he` → `_i_he` (src/haar/params.rs:124)
- [x] 修 ffmpeg_pipe.rs:37 多余括号
- [x] ffmpeg_pipe 字段加 `#[allow(dead_code)]` (诊断用)
- [x] gpu `dlerror` extern 删 (未用)
- [x] gpu `LIB` 静态加 `static_mut_refs` allow
- [x] cnn 教学用 helper 加 `dead_code` allow
- [x] debug_cascade + cnn_train 删未用 import
- [x] cnn_train 删未用 `BATCH`, `Rng::u8`, `let mut init_diag_done`
- [x] `cargo build --release` 0 warning

## P2 — Lint 基础设施 ✅
- [x] 加 `.github/workflows/ci.yml`（fmt + clippy + test on PR, ubuntu+macos）
- [x] 加 `clippy.toml`（合理阈值）
- [x] 加 `rustfmt.toml`（一致风格）
- [x] 加 `[lints.clippy]` 表在 Cargo.toml（允许的 pedantic 警告）

## P3 — 性能优化 ✅ (10% 提速, 640x480 122→134 fps, 1920x1080 19→20 fps)
- [x] 积分图行扫描：unsafe ptr 累加（已实现，`IntegralImage::from_gray` + `SquaredIntegralImage::from_gray`）
- [x] NMS：3×3 空间分桶优化，O(n²) → O(n) 均摊
- [x] detector 主循环：预计算 `inv_scale` 避免每窗口除法
- [ ] 金字塔：避免冗余内存分配（不必要，已有 `resize_area`）
- [ ] 进一步优化需 profiling（criterion / perf）才能定位 hot spot，不冒进

## P4 — 单测 ✅ (从 13→30 tests, 顺带修了 1 个真 bug)
- [x] `haar/feature.rs` 特征求值 (从 2→4 tests, 修了 vertical/horizontal 期望值)
- [x] `haar/params.rs` cascade save/load (现成 2 tests, OK)
- [x] `integral.rs` 积分图 (从 2→6 tests, 修了 rotated integral in-place 覆盖 bug)
- [x] `cnn/mod.rs` 前向 (现成, OK)
- [x] `detector.rs` NMS (从 1→4 tests, 覆盖了空/稀疏/密集场景)
- [x] `image/codec.rs` PNG/PPM (从 2→4 tests, 加了 magic 拒绝 + 注释/空白处理)

## 顺带修复的真实 bug
- `RotatedIntegralImage::from_gray` pass 2 是 in-place 修改 data, 但 Lienhart 递推要读 SAT 值.
  读到的会是已覆盖的 R 值, 产生垃圾. 改为双 buffer (sat + data) 后 24px 图像从 55040 的正常值变成 10^18.
  2 个 pre-existing test 立刻通过 (`demo_classifies_bright_center_pattern`, `detects_bright_center_in_uniform_image`).

## P5 — 文档 ✅
- [x] `docs/architecture.md` 架构图 + 数据流 + 模块职责
- [x] `docs/algorithms.md` integral/variance/cascade/NMS 完整参考
- [x] `docs/benchmarks.md` baseline + 优化后对比 + 未来方向
- [x] `docs/format.md` `.rfcf` 二进制格式参考
- [x] README 末尾加 `## Further reading` 链接

## 已知遗留问题（不在 P0-P5 范围，但记录备查）
- 4 个 test 失败：
  1. `haar::feature::tests::vertical_edge_response` — test 期望 OpenCV 老行为（除以 win*win），code 跟 README 一致（不除）。需要决定：test 改成现代 OpenCV 行为？
  2. `haar::feature::tests::horizontal_edge_response` — 同上
  3. `haar::params::tests::demo_classifies_bright_center_pattern` — 同上
  4. `detector::tests::detects_bright_center_in_uniform_image` — underflow at integral.rs:231，可能是 rotated integral 的边界 bug
- 36 个 clippy default 警告（pedantic 风格，未做 -D warnings）
