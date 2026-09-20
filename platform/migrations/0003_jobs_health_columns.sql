-- 2026-09-20 加:jobs 表健康跟踪列。
--
-- 目的:
-- - `started_at` / `finished_at` 已经以毫秒存在,这里新增 wall-clock
--   (秒级,DB 端时间)用于跨重启分析与 orphan 检测。
-- - `heartbeat_at` 每 30s 由 run_job 心跳线程写入一次;启动时 scan
--   `status='running' AND heartbeat_at < now()-300s` 的行并标 error,
--   解决"服务器在 running 中途挂掉、PG 留下 stuck 'running' 行"的
--   僵尸任务问题。
-- - `updated_at` 通用审计列,任何 UPDATE 都刷新。
-- - `archived` 把内存里已有的归档标志(见 `Job::archived`)落到 DB,
--   让 PG 端的"列出非归档任务"查询走索引,不再需要 join 内存。
--
-- 设计取舍:
-- - 所有列 IF NOT EXISTS,兼容老库(2026-08-20 平台 v0.1 起的库)。
-- - 索引用 BRIN 而非 B-tree:`updated_at` / `heartbeat_at` 是单调递增
--   时间戳,典型查询是"最近 N 分钟未心跳";BRIN 在这类查询上比 B-tree
--   省 99% 空间、近似等效的 seek 性能。
-- - 状态枚举检查约束已在 0001 加,本迁移不重复。
-- - `error_code` 列(text)新增:与 API 层 `error_code` 字段对齐,便于
--   按错误码聚合分析("哪些任务因为 queue_full / cascade_missing 失败")。

ALTER TABLE jobs ADD COLUMN IF NOT EXISTS started_at    TIMESTAMPTZ;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS heartbeat_at  TIMESTAMPTZ;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS updated_at    TIMESTAMPTZ;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS archived      BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS error_code    TEXT;

-- BRIN 索引:针对按时间区间扫描的查询(`heartbeat_at < now() - interval`,
-- `created_at > ...`)。BRIN 占用极小(~kB 级),对大数据集(>10w 行)有
-- 显著的 seek 优势。
CREATE INDEX IF NOT EXISTS jobs_heartbeat_brin
    ON jobs USING BRIN (heartbeat_at) WITH (pages_per_range = 32);
CREATE INDEX IF NOT EXISTS jobs_updated_brin
    ON jobs USING BRIN (updated_at) WITH (pages_per_range = 32);

-- 状态 B-tree 索引:`status='running' AND heartbeat_at < ...` 这类
-- orphan 扫描会用到。BRIN 不擅长单值等值查询,这里加 B-tree 补齐。
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status);

-- 组合索引:列已归档任务时按时间倒序查。左侧 archived 让"隐藏归档"
-- 的查询(where archived=false)走 index-only scan。
CREATE INDEX IF NOT EXISTS jobs_archived_created_idx
    ON jobs(archived, created_ms DESC);

-- `error_code` 索引:让"按错误码聚合"类查询(运维 dashboard)走索引,
-- 而不是全表扫。
CREATE INDEX IF NOT EXISTS jobs_error_code_idx ON jobs(error_code);