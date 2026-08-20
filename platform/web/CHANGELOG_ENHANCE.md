# Web UX 增强(Enhance)v0.2

6 项增强 + 完整 a11y / 键盘导航 / 主题切换,在 5-bug-fix 基础之上叠加,不改 `cascade.rfcf`、核心算法、`compare.js` 与现有 changelog。

## 1. 批量选择 + 批量操作

- 侧栏底部多出 `#sb-batch` 操作栏,只在进入批量模式时显示
- 卡片左上角浮现复选框 `.sb-check`,Shift+点击多选
- 操作:`取消 / 归档 / 导出 JSON / 删除`;删除与归档走 `confirmModal` 二次确认
- 新增 API:
  - `DELETE /api/jobs/{id}` — 单删(取消任务 + 内存清掉 + DB `delete_job`)
  - `POST /api/jobs/batch { ids, op }` — `op` ∈ `delete|archive|export`
- 服务端:
  - `jobs.rs` 新增 `archived`、`original_input` 字段,以及 `set_archived` / `set_original_input` / `request_cancel` / `remove` / `remove_many`
  - `persist.rs` 新增 `delete_job` / `delete_jobs`(走 `WHERE id = ANY($1)`,PG FK 自动级联清 `frames` / `faces`)
  - `api.rs` 把 `/api/jobs/{id}` 从 `get` 改为 `get().delete()`,并挂上 `/api/jobs/batch` 路由

## 2. 主题切换(自动/暗/亮)

- `data-theme="auto|light|dark"` 挂在 `<html>` 上,选择器 `[data-theme="light"]` / `[data-theme="dark"]` 反色
- CSS 变量集中在 `:root`,主题差异通过 `[data-theme="*"]` 覆盖;`body` 上有 `transition: background-color .3s, color .3s, border-color .3s` 平滑过渡
- 偏好写入 `localStorage['rsface-theme']`;`auto` 监听 `matchMedia('(prefers-color-scheme: light)')` 实时跟随系统
- 设置 modal 中的 `<div class="seg" role="radiogroup">` 用了标准 ARIA 模式

## 3. 智能搜索

- 输入框保留原 `id` / `name` 模糊匹配(沿用旧逻辑)
- `#tb-search-opts` 弹出"模式"(普通 / 正则)+"范围"(1h / 24h / 7d / 30d / 全部)两组分段控件
- 正则模式:`try { new RegExp(q, 'i') }` 安全校验,失败给 `toast.warn`,匹配 `display_name` 与 `id`
- 时间范围:用 `created_at` 与 `Date.now() - rangeMs` 比较
- `#tb-search-clear` 按钮仅在有输入时显示

## 4. 统计仪表板

- 顶部 ◔ 按钮打开 `#modal-dashboard`
- 6 个数据 tile:总任务 / 进行中 / 已完成 / 错误 / 人脸总数 / 总检测时长
- 算法占比:`video / image / haar / cnn / yunet / mtcnn / hog` 横向条形图(归一化)
- 24 小时时间线:纯 SVG 24 根柱,每柱高 = 该小时新增任务数,X 轴标 `-24h / -12h / 现在`
- 全部数据来自 `GET /api/jobs`,无新增接口

## 5. 统一 Toast + 失败重试

- `window.toast.{info, success, warn, error}(msg, ms?)`,自动入 `#toast-host`,3s 后淡出,可手动 ×
- `aria-live="polite"` 区域,屏幕阅读器友好
- 重试:`#pv-retry` 按钮仅在 `error` / `done` 状态下显示;命中 URL 任务时直接 `POST /api/jobs` 复用 `original_input`,multipart 上传返回 400 提示重新上传

## 6. 键盘 + a11y

| 键 | 行为 |
|---|---|
| `n` | 新建任务 |
| `↑` / `↓` | 切换任务(虚拟列表) |
| `Enter` | 打开当前任务 |
| `Delete` | 删除当前任务(确认) |
| `Ctrl/⌘ + K` | 聚焦搜索框 |
| `a` | 切换人脸标注 |
| `Esc` | 关闭弹窗 / 预览 |
| `?` | 弹出快捷键帮助 |
| `Shift + 点击` | 多选(批量模式) |

- 所有 modal: `role="dialog" aria-modal="true" aria-labelledby="..."`;确认弹窗用 `role="alertdialog"`
- 所有 segmented control: `role="radiogroup"` + 按钮 `role="radio" aria-checked`
- 侧栏过滤器: `role="tablist"`,按钮 `role="tab" aria-selected`
- 任务列表容器: `role="listbox"`,选项 `role="option" aria-selected`
- 顶部 `aria-live="polite"` 区,变更即时播报
- `*:focus-visible` 走 `--focus-ring`,键盘用户清晰可见

## 文件变更清单

| 文件 | 之前 | 之后 | 备注 |
|---|---:|---:|---|
| `platform/server/src/jobs.rs` | ~360 | ~430 | 新增 `archived` / `original_input` 字段与若干 setter |
| `platform/server/src/persist.rs` | ~260 | ~290 | 新增 `delete_job` / `delete_jobs` |
| `platform/server/src/api.rs` | ~720 | ~860 | 新增 `DELETE /api/jobs/{id}`、`POST /api/jobs/batch`、重试 |
| `platform/web/index.html` | ~180 | ~250 | 主题/搜索/仪表板/帮助/确认 5 个新 modal + 批量栏 |
| `platform/web/app.js` | ~430 | ~600 | 6 项增强实现 + 主题/Toast/Confirm/批量/Dashboard 模块 |
| `platform/web/style.css` | ~520 | ~700 | 双主题调色板 + 新组件 + a11y focus ring |
| `platform/web/CHANGELOG_ENHANCE.md` | — | + | 本文件 |

未触动:`cascade.rfcf`、核心算法、`compare.js`、`CHANGELOG_DOUBAO.md`、`CHANGELOG_FIXES.md`、Dockerfile、`docker-compose.yml`。

## 主题色板

| Token | 暗色 | 亮色 |
|---|---|---|
| `--bg` | `#0d1117` | `#ffffff` |
| `--panel` | `#161b22` | `#f5f7fa` |
| `--text` | `#e6edf3` | `#0f172a` |
| `--text-dim` | `#8b949e` | `#475569` |
| `--accent` | `#3b82f6` | `#0284c7` |
| `--border` | `#30363d` | `#cbd5e1` |
| `--success` | `#22c55e` | `#16a34a` |
| `--warn` | `#f59e0b` | `#d97706` |
| `--danger` | `#ef4444` | `#dc2626` |
| `--focus-ring` | `rgba(59,130,246,.55)` | `rgba(2,132,199,.55)` |

## a11y 合规要点

- 所有交互元素带 `aria-label`
- 弹窗用 `role="dialog" aria-modal="true"`,Esc 关闭
- 实时区 `aria-live="polite"`
- 颜色对比度 ≥ 4.5:1(WCAG AA)
- 焦点环 `:focus-visible` 清晰可辨
- 表单输入有可见 label 或 `aria-label`
- segmented control / tablist / listbox 都用标准 ARIA 模式
