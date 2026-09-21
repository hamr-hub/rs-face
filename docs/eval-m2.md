# M2 活体检测 FAR/FRR 实测报告

- 评测日期：2026-09-21
- 被测代码：`src/liveness.rs`（纯预处理/决策）+ `src/liveness_detector.rs`（MiniFASNet ONNX），提交基线 `25dbbc9`，工作树 `dca0246`
- 评测方式：独立 Python 脚本，**未修改任何 `src/` 代码**，与库内完全相同的预处理与融合规则（80×80、原始 `[0,255]` BGR、双头 softmax 平均、argmax==real 且 `p_real ≥ 阈值`）
- 运行设备：Jetson（8GB），CPU 单进程小批量（评测时未占用 GPU，避免与并行任务争内存）

## 1. 结论（先说结果）

| 指标 | 验收要求 | 实测 | 是否达标 |
|---|---|---|---|
| 真人样本数 | ≥100 | **179** | ✅ |
| 攻击样本数 | ≥100 | **540**（打印 360 + 屏幕 180） | ✅ |
| FAR（攻击放行率） | ≤1% | 与 FRR 无法同时满足（见下） | ❌ 联合不达标 |
| FRR（真人被拒率） | ≤5% | 同上 | ❌ 联合不达标 |
| 阈值 + ROC | 必须给出 | 见 §4/§5 | ✅ |

**关键数字（部署规则 = argmax==real 且 p_real ≥ t）：**

| 工作点 | 阈值 t | FAR | FRR | FAR-打印 | FAR-屏幕 |
|---|---|---|---|---|---|
| 纯 argmax（当前默认 `min_real_score=0.0`） | 0.0000 | **2.04%**（11/540） | **12.29%**（22/179） | 2.50% | 1.11% |
| Youden 最佳（原始分数） | 0.4413 | 1.67%（9/540） | 12.29%（22/179） | 2.22% | 0.56% |
| 满足 FAR≤1% 的最低 FRR | 0.5918 | 0.93%（5/540） | **18.99%**（34/179） | 1.11% | 0.56% |
| 满足 FRR≤5% 的最低 FAR | — | **不存在**（FRR 地板为 12.29%） | — | — | — |

- 原始 `p_real` 分数的 **ROC AUC = 0.9877，EER ≈ 5.6%**——模型本身区分度良好，但当前 ONNX 权重在该测试集上的**最佳联合工作点仍不满足 FAR≤1% 且 FRR≤5%**。
- 瓶颈在 FRR：有 22/179 真人（12.3%）连 argmax 都不是 real 类（15 条被判 screen、7 条被判 paper），抬高阈值只能降 FAR，无法救这些真人，故 FRR≤5% 对任何阈值都不可达。

## 2. 样本构成

- 数据集：CASIA-FASD 测试帧的 Hugging Face 镜像 `akahana/anti-spoofing-casiafasd`（`casiafasd.tar.gz`，69.3 MB，Apache 可下载；标签取自文件名 `_real/_fake` 后缀）
- 解压路径：`data/eval_m2/test_img/test_img/color/`（4816 个原始 jpg；256×256，由测试视频抽帧并 JPEG 重编码）
- 30 名受试者；真人仅来自 genuine 视频类型（token `1 / 2 / HR_1`），攻击按 CASIA 类型细分：
  - 打印攻击（扭曲照片/裁剪照片）：token `3/4/5/6/HR_2/HR_3`
  - 屏幕回放攻击：token `7/8/HR_4`
- 去相关抽样：同一源视频（共 360 个）最多抽 2 帧（`--per-source 2`，固定随机种子 42），避免同一视频相邻帧虚增样本量。
- 最终：真人 179、打印 360、屏幕 180。
- 人脸检测：固定使用仓库 pinned 的 YuNet ONNX（`models/face_detection_yunet_2023mar.onnx`，sha256 已验证），719/719 全部检出（0 次居中兜底）。

## 3. 权重与推理可用性

`tools/fetch_models.sh --verify-only` 三个权重全部 sha256 校验通过：

- `face_detection_yunet_2023mar.onnx` ✅
- `2.7_80x80_MiniFASNetV2.onnx` ✅
- `4_0_0_80x80_MiniFASNetV1SE.onnx` ✅

独立冒烟脚本 `tools/eval_m2_smoke.py` 实跑双头 forward（zero/random probe 均输出合法三类概率），确认推理真实可跑、非桩实现。

## 4. 阈值分析

- 决策语义与 `src/liveness.rs::decide` 一致：两个 crop scale（2.7× / 4.0×）各自 softmax，平均后 argmax==real **且** real 概率 ≥ `min_real_score` 才放行。
- argmax 规则下 FPR 的上限被"攻击中 argmax==real 的比例"钉死在 2.04%；要压到 FAR≤1% 必须把阈值提到 0.59，代价是 FRR 升至 19%。
- 22 个 argmax 即误判的真人集中在受试者 12/13/14/15/17/19 等，典型如 `13_1.avi_175_real.jpg`（probs 0.001/0.030/0.970，被强判 screen），说明这些低质量/特定光照真人帧落在了攻击纹理分布内。

## 5. ROC 曲线数据

- 部署规则 ROC（threshold, FAR, FRR, TPR, FAR_print, FAR_screen，719 样本逐阈值扫描）：
  **`data/eval_m2/m2_roc.csv`**
- 逐样本分数（路径、标签、攻击子类、YuNet 是否命中、三类概率）：
  **`data/eval_m2/m2_scores.csv`**
- 摘要指标（原始分数）：AUC=0.9877，EER≈0.0557。

## 6. 复跑命令

```bash
# 1. 权重（已在 models/ 且 sha256 通过）
bash tools/fetch_models.sh --verify-only

# 2. 数据（约 70MB，解压到 data/eval_m2/）
mkdir -p data/eval_m2 && cd data/eval_m2
curl -fLO https://huggingface.co/datasets/akahana/anti-spoofing-casiafasd/resolve/main/casiafasd.tar.gz
tar xzf casiafasd.tar.gz
cd ../..

# 3. 冒烟 + 正式评测（CPU，单进程）
python3 tools/eval_m2_smoke.py
python3 tools/eval_m2_far_frr.py
```

依赖：`onnxruntime`、`opencv-python-headless`、`numpy`。

## 7. 未达标原因分析

1. **样本域差异**：本次为 256×256、JPEG 重编码的 CASIA 抽帧；MiniFASNet 上游主要在其内部高清采集域上标定，低分辨率 + 重编码改变了其依赖的纹理/Moiré 统计，真人低分帧增多。
2. **真人侧光照/质量长尾**：FRR 全部来自被强判 screen/paper 的真人，阈值对此无能为力，属于模型/域适配问题而非阈值问题。
3. **默认阈值（0.0）即纯 argmax**：在该测试集上本身就有 2% FAR 与 12% FRR，不满足 M2。

## 8. 后续建议（不在本任务实施）

- 用部署目标摄像头采集真人与实拍打印/屏幕样本做域内评测（≥100/类），优先复测 FRR；
- 在域内数据上对 MiniFASNet 做小规模微调，或加入人脸质量/光照过滤前置；
- 分数校准（temperature/Platt）后再定阈值。

## 9. 活体门控接入 1:1 / 1:N / 注册的接口建议（只给建议，不改代码）

现有库接口已具备门控所需全部输入，无需改算法核心，建议在 platform 服务层包一层可配置策略：

- 配置项（建议挂在 server 配置）：
  - `liveness.enabled: bool`（总开关，默认开）
  - `liveness.threshold: f32`（即 `LivenessConfig.min_real_score`，本评测数据下 0.44≈Youden、0.59≈FAR1%；上线阈值以域内复测为准）
  - `liveness.mode: {enforce, log_only}`（enforce 拦截，log_only 仅记录分数灰度上线）
- 调用契约：检测 → 对**最大脸**（或每张脸）调用 `LivenessDetector::check(img, det) -> LivenessOutcome`，取 `is_real / real_score / label()`。
- 三处接入点建议：
  1. **注册/入库**：活体失败直接拒绝建档（防止把攻击样本录进库），返回明确错误码（如 `409 LIVENESS_FAILED`）并附 `real_score`。
  2. **1:1 比对**：先活体后比对；活体不过直接短路返回"未通过活体"，不再消耗比对并返回否定结果。
  3. **1:N 识别**：仅对 `is_real` 的人脸送入识别候选；攻击脸只画框不返回身份，防止用照片冒认。
- 可观测性：每次门控记录 `real_score / 三类 probs / 判定 / mode`，便于上线后按真实分布回调阈值；建议加连续 N 帧多数表决以降低视频流 FRR。

## 10. 证据与产物路径

| 内容 | 路径 |
|---|---|
| 冒烟脚本 | `tools/eval_m2_smoke.py` |
| FAR/FRR 评测脚本 | `tools/eval_m2_far_frr.py` |
| ROC 数据 | `data/eval_m2/m2_roc.csv` |
| 逐样本分数 | `data/eval_m2/m2_scores.csv` |
| 数据集压缩包/解压帧 | `data/eval_m2/casiafasd.tar.gz`、`data/eval_m2/test_img/` |
