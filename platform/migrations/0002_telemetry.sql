-- 2026-09-18 加:用户行为埋点表。
--
-- 设计目标:
-- - 极简 schema,只保留维度 + 不太活跃(名字 / 时间 / 会话 / 页面 / props JSONB)。
-- - 不建索引:默认按 created_at DESC 顺序扫描,体量小(每日 1w-100w 行),
--   真要查也走 BRIN 或 weekly partition,这里先不优化。
-- - props JSONB:前端送任意结构;服务端 `is_safe_event` 已过滤敏感字段,
--   但 props 仍可能含业务数据,所以用 JSONB 而不是 JSON,后续可以用 jsonb_path
--   查询 / GIN 索引。
--
-- 没有 retention / vacuum 策略:留给运维周期归档(超过 30 天 DELETE 掉)。

CREATE TABLE IF NOT EXISTS telemetry (
    id        BIGSERIAL PRIMARY KEY,
    name      TEXT        NOT NULL,
    ts_ms     BIGINT      NOT NULL,
    received_ms BIGINT    NOT NULL,
    session   TEXT,
    path      TEXT,
    props     JSONB
);