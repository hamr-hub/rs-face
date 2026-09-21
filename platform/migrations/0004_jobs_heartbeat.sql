-- 2026-09-21 加:job worker 心跳列 + 死锁 janitor 支持。
--
-- 背景:
-- - 0003 加了 `heartbeat_at` 给启动时的 orphan reaper 用(阈值 5 分钟,只在
--   server restart 时跑一次)。这次的 `last_heartbeat_ts` 走另一个时间轴:
--   进程内运行时周期(5 秒)刷新,janitor 任务每 30 秒扫描
--   `status='running' AND last_heartbeat_ts < now() - 30s` 的行,
--   标 failed + reason='worker_died'。
-- - 区分列名(`heartbeat_at` vs `last_heartbeat_ts`)是两个机制并存,职责清晰:
--   * `heartbeat_at`   — 启动 orphan reaper,容错窗口大(5 min)
--   * `last_heartbeat_ts` — 运行时 janitor,容错窗口小(30 s),针对
--     worker 进程活着但线程卡死 / 静默失败(panic / OOM / deadlock)的情况。
--
-- backfill:
-- - 现有 running 行的 `last_heartbeat_ts` 一律初始化为 `started_at`(若有)
--   或 `updated_at`(刚 mark_started 的行 heartbeat 还没写),`started_at` 为
--   null 的老库行直接 `now()`(它们大概率早就是 zombie,被 janitor 标 failed
--   后运维会看到具体原因)。
-- - 已有 done/cancelled/error 行不影响(janitor 只看 status='running')。
--
-- 设计取舍:
-- - 新增独立列不复用 `heartbeat_at`,避免两个时间语义在同一个字段上叠加
--   (读侧分不清"老 orphan reaper 用的"vs"新 janitor 用的")。
-- - 索引用 BRIN(单调时间戳,小数据量足够);查询模式固定为
--   `WHERE status='running' AND last_heartbeat_ts < now() - interval`,
--   status 已有 B-tree 索引,组合即可。
-- - 不加 CHECK / NOT NULL:NULL 表示"还没开始心跳",janitor 跳过即可。

ALTER TABLE jobs ADD COLUMN IF NOT EXISTS last_heartbeat_ts TIMESTAMPTZ;

-- backfill:为历史 running 行填一个非 NULL 时间戳,避免 janitor 第一轮
-- 误判为"30s 内没心跳"。started_at 已写(0003 加)的行用它;没有 started_at
-- 的老库行用 updated_at;都没有(理论上不应该发生)兜底 now()。
UPDATE jobs
   SET last_heartbeat_ts = COALESCE(started_at, updated_at, now())
 WHERE last_heartbeat_ts IS NULL;

-- 索引:janitor 查询 = `status='running' AND last_heartbeat_ts < now()-interval`;
-- status 已 B-tree(last_heartbeat_ts 过滤后行数极少,直接 seq scan 也 OK)。
-- 这里只补 BRIN,后续真实生产数据大了再补组合索引。
CREATE INDEX IF NOT EXISTS jobs_last_heartbeat_brin
    ON jobs USING BRIN (last_heartbeat_ts) WITH (pages_per_range = 32);
