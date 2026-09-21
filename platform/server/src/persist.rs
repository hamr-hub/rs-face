//! PostgreSQL 持久化层:job/frame/face 三张表。
//!
//! - DB 不可用时 JobRegistry 降级为纯内存(不报错),保留可用性;
//! - 所有写入走 `persist_*` 函数,无返回值(void 风格)。

use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;

use crate::jobs::{FaceEntry, FrameResult, JobKind, JobStats, JobStatus};

/// 前端埋点单条事件的扁平化形态(由 api::telemetry_ingest 接收并转发)。
/// 字段和 `api::TelemetryEvent` 一致;这里独立定义是为了让 persist 层
/// 不依赖 axum 提取器类型(便于单测和将来加其它 ingest 路径)。
#[derive(Clone, Debug, serde::Deserialize)]
pub struct TelemetryEvent {
    pub name: String,
    pub ts: u64,
    #[serde(default)]
    pub props: serde_json::Value,
    #[serde(default)]
    pub session: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
}

#[derive(Clone)]
pub struct Db {
    pub pool: Option<PgPool>, // None 时降级为不持久化
}

impl Db {
    pub async fn connect(url: &str) -> Self {
        match PgPoolOptions::new()
            .max_connections(8)
            .acquire_timeout(std::time::Duration::from_secs(5))
            .connect(url)
            .await
        {
            Ok(pool) => {
                tracing::warn!("[persist] connected to PostgreSQL");
                Self { pool: Some(pool) }
            }
            Err(e) => {
                tracing::warn!("[persist] PG connect failed: {e} — running in memory-only mode");
                Self { pool: None }
            }
        }
    }

    /// 按文件名升序 apply 尚未记录的迁移,返回本次新应用的文件数。
    ///
    /// 每个文件整体放在一个事务里执行并在成功后写入 `schema_migrations`:
    /// - 已应用的文件不再重跑(老逻辑每次启动全量重跑,依赖每条 DDL 幂等);
    /// - 任何一条语句失败 → 整个文件回滚并把错误返回给调用方,不会留下
    ///   半应用 schema(PG 里事务出错后后续语句也无法执行);
    /// - 调用方对错误做 fail-fast,而不是像"PG 连不上"那样降级内存模式:
    ///   对着一个残缺 schema 静默丢持久化比启动失败更危险。
    pub async fn migrate(&self) -> Result<u64, String> {
        let Some(pool) = &self.pool else {
            return Ok(0);
        };
        // 追踪表自身保持幂等、不纳入追踪(鸡生蛋问题)。
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS schema_migrations (\n\
             filename   TEXT PRIMARY KEY,\n\
             applied_ms BIGINT NOT NULL\n\
             )",
        )
        .execute(pool)
        .await
        .map_err(|e| format!("ensure schema_migrations table: {e}"))?;

        let files: &[(&str, &str)] = &[
            (
                "0001_init.sql",
                include_str!("../../migrations/0001_init.sql"),
            ),
            (
                "0002_telemetry.sql",
                include_str!("../../migrations/0002_telemetry.sql"),
            ),
            (
                "0003_jobs_health_columns.sql",
                include_str!("../../migrations/0003_jobs_health_columns.sql"),
            ),
            (
                "0004_jobs_heartbeat.sql",
                include_str!("../../migrations/0004_jobs_heartbeat.sql"),
            ),
        ];
        let mut applied = 0u64;
        for (name, sql) in files {
            let already: bool = sqlx::query_scalar(
                "SELECT EXISTS(SELECT 1 FROM schema_migrations WHERE filename = $1)",
            )
            .bind(*name)
            .fetch_one(pool)
            .await
            .map_err(|e| format!("check applied migration {name}: {e}"))?;
            if already {
                continue;
            }
            let mut tx = pool
                .begin()
                .await
                .map_err(|e| format!("begin tx for {name}: {e}"))?;
            // sqlx 扩展协议不支持多语句;按顶层 `;` 切分,切分器识别
            // 注释 / 字符串 / 美元引用,不会被字面量里的 `;` 误切。
            for stmt in split_sql_statements(sql) {
                sqlx::query(&stmt)
                    .execute(&mut *tx)
                    .await
                    .map_err(|e| format!("migration {name} statement failed: {e}\n>> {stmt}"))?;
            }
            sqlx::query("INSERT INTO schema_migrations(filename, applied_ms) VALUES ($1, $2)")
                .bind(*name)
                .bind(now_ms_u64() as i64)
                .execute(&mut *tx)
                .await
                .map_err(|e| format!("record migration {name}: {e}"))?;
            tx.commit()
                .await
                .map_err(|e| format!("commit migration {name}: {e}"))?;
            applied += 1;
            tracing::warn!("[persist] applied migration {name}");
        }
        Ok(applied)
    }

    #[allow(dead_code)] // 保留:DB 恢复 / 健康自检 API 面
    pub fn is_enabled(&self) -> bool {
        self.pool.is_some()
    }

    /// 关闭连接池(优雅停机时调用,让 PG 侧的 in-flight 写入落地)。
    /// sqlx PgPool 是引用计数的;clone 只是 Arc 增减。真正关闭由
    /// `Pool::close()` 广播,后续 acquire 会失败 — 进程退出路径专用。
    pub async fn close(&self) {
        if let Some(pool) = &self.pool {
            pool.close().await;
        }
    }

    pub async fn insert_telemetry_batch(&self, events: &[TelemetryEvent]) {
        if events.is_empty() {
            return;
        }
        let Some(pool) = &self.pool else {
            return;
        };
        let n = events.len();
        let mut name = Vec::with_capacity(n);
        let mut ts = Vec::with_capacity(n);
        let mut now_ms = Vec::with_capacity(n);
        let mut session = Vec::with_capacity(n);
        let mut path = Vec::with_capacity(n);
        let mut props = Vec::with_capacity(n);
        let received = now_ms_u64();
        for e in events {
            name.push(e.name.clone());
            ts.push(e.ts as i64);
            now_ms.push(received as i64);
            session.push(e.session.clone());
            // path 截断防滥用:超过 256 直接丢(异常路径 / 错误数据)。
            path.push(
                e.path
                    .as_ref()
                    .map(|p| if p.len() > 256 { "" } else { p.as_str() })
                    .unwrap_or("")
                    .to_string(),
            );
            props.push(e.props.clone());
        }
        let res = sqlx::query(
            "INSERT INTO telemetry (name, ts_ms, received_ms, session, path, props)
             SELECT * FROM UNNEST($1::text[], $2::bigint[], $3::bigint[], $4::text[], $5::text[], $6::jsonb[])"
        )
        .bind(&name)
        .bind(&ts)
        .bind(&now_ms)
        .bind(&session)
        .bind(&path)
        .bind(&props)
        .execute(pool)
        .await;
        if let Err(e) = res {
            tracing::warn!("[persist] insert_telemetry_batch failed: {e}");
        }
    }

    pub async fn insert_job(
        &self,
        id: &str,
        kind: JobKind,
        display_name: &str,
        status: JobStatus,
        created_ms: u64,
    ) {
        let Some(pool) = &self.pool else {
            return;
        };
        let kind = match kind {
            JobKind::Image => "image",
            JobKind::Video => "video",
            JobKind::Stream => "stream",
        };
        let status = status_to_str(status);
        let _ = sqlx::query(
            "INSERT INTO jobs (id, kind, display_name, status, created_ms) VALUES ($1,$2,$3,$4,$5)
             ON CONFLICT (id) DO NOTHING",
        )
        .bind(id)
        .bind(kind)
        .bind(display_name)
        .bind(status)
        .bind(created_ms as i64)
        .execute(pool)
        .await;
    }

    /// 记录任务实际使用的算法(在 run_job 构建 detector 后调用)。
    pub async fn set_algo(&self, id: &str, algo: &str) {
        let Some(pool) = &self.pool else {
            return;
        };
        let _ = sqlx::query("UPDATE jobs SET algo=$2 WHERE id=$1")
            .bind(id)
            .bind(algo)
            .execute(pool)
            .await;
    }

    pub async fn update_job_status(
        &self,
        id: &str,
        status: JobStatus,
        finished_ms: Option<u64>,
        error: Option<&str>,
    ) {
        let Some(pool) = &self.pool else {
            return;
        };
        let s = status_to_str(status);
        let _ = sqlx::query("UPDATE jobs SET status=$2, finished_ms=$3, error=$4 WHERE id=$1")
            .bind(id)
            .bind(s)
            .bind(finished_ms.map(|v| v as i64))
            .bind(error)
            .execute(pool)
            .await;
    }

    pub async fn set_original_key(&self, id: &str, key: &str) {
        let Some(pool) = &self.pool else {
            return;
        };
        let _ = sqlx::query("UPDATE jobs SET original_key=$2 WHERE id=$1")
            .bind(id)
            .bind(key)
            .execute(pool)
            .await;
    }

    /// 标记 job 进入 running 时刻(用于 DB 端分析 + orphan 检测的起点)。
    /// 幂等:同 id 重复调用覆盖一次。失败只打日志(persist 是尽力而为)。
    pub async fn mark_started(&self, id: &str, started_ms: u64) {
        let Some(pool) = &self.pool else {
            return;
        };
        let _ = sqlx::query(
            "UPDATE jobs SET status='running', started_at=to_timestamp($2::bigint / 1000.0), \
             heartbeat_at=now(), updated_at=now() WHERE id=$1",
        )
        .bind(id)
        .bind(started_ms as i64)
        .execute(pool)
        .await;
    }

    /// heartbeat:run_job 内的 watchdog 心跳线程每 30s 调用一次,
    /// 让 DB 知道这个 job 还在跑。orphan 扫描以
    /// `heartbeat_at < now() - 5min` 判定僵尸。
    pub async fn heartbeat(&self, id: &str) {
        let Some(pool) = &self.pool else {
            return;
        };
        let _ = sqlx::query(
            "UPDATE jobs SET heartbeat_at=now(), updated_at=now() WHERE id=$1 AND status='running'",
        )
        .bind(id)
        .execute(pool)
        .await;
    }

    /// 高频心跳:worker 主循环每 ~5s 写一次 `last_heartbeat_ts = now()`。
    /// 与 `heartbeat()`(30s 周期,服务于启动 orphan reaper)语义不同:
    /// 这里走更细的粒度,让运行时 janitor(30s 扫一次)能区分"worker 真死"
    /// vs"worker 还在跑但当前帧耗时长"。
    /// 仅写 running 行;job 已终止时 UPDATE 影响 0 行,无害。
    pub async fn heartbeat_realtime(&self, id: &str) {
        let Some(pool) = &self.pool else {
            return;
        };
        let _ = sqlx::query(
            "UPDATE jobs SET last_heartbeat_ts=now(), updated_at=now() \
             WHERE id=$1 AND status='running'",
        )
        .bind(id)
        .execute(pool)
        .await;
    }

    /// 死锁 janitor:扫描所有 `status='running' AND last_heartbeat_ts < now() - stale_secs`
    /// 的行,把它们的 status 改为 'error',error 字段填 `worker_died` 作为
    /// 可读原因。返回被标记的行数。
    ///
    /// **触发场景**:worker 线程被 panic / OOM / deadlock / runtime 强杀,
    /// 但进程还在跑 / 重新拉起了新进程,DB 留下 status='running' 但
    /// 心跳不再刷新的行。前端看不到这种 zombie,UI 永远卡在"运行中"。
    ///
    /// **与 `reap_orphans` 的区别**:reap_orphans 在 server 启动时跑一次,
    /// 阈值 5 分钟,负责"上次进程崩溃留下的";这里是运行时周期任务,
    /// 阈值 30 秒,负责"本次运行期 worker 静默死了"。
    pub async fn janitor_kill_stale(&self, stale_secs: i64) -> Option<u64> {
        let pool = self.pool.as_ref()?;
        let res = sqlx::query(
            "UPDATE jobs SET status='error', \
             error='worker_died', \
             error_code='worker_died', \
             finished_ms=$2, \
             updated_at=now() \
             WHERE status='running' \
               AND last_heartbeat_ts IS NOT NULL \
               AND last_heartbeat_ts < now() - make_interval(secs => $1)",
        )
        .bind(stale_secs as f64)
        .bind(now_ms_u64() as i64)
        .execute(pool)
        .await;
        match res {
            Ok(r) => Some(r.rows_affected()),
            Err(e) => {
                tracing::warn!("[persist] janitor_kill_stale failed: {e}");
                None
            }
        }
    }

    /// 设置归档标志(侧栏默认隐藏;与内存 `Job::archived` 字段镜像)。
    pub async fn set_archived(&self, id: &str, archived: bool) {
        let Some(pool) = &self.pool else {
            return;
        };
        let _ = sqlx::query("UPDATE jobs SET archived=$2, updated_at=now() WHERE id=$1")
            .bind(id)
            .bind(archived)
            .execute(pool)
            .await;
    }

    /// 设置错误码(API `error_code` 字段):与 `error` 文本共存,
    /// 便于 PG 端按 error_code 聚合("哪些 task 因 queue_full 失败")。
    /// 保留为持久化 API 表面,jobs.rs 在 run_job 失败时调用;
    /// 当前编译期只有 reap_orphans 内部用,标 `allow(dead_code)` 避免
    /// 暂时未被引用时告警。
    #[allow(dead_code)]
    pub async fn set_error_code(&self, id: &str, code: &str) {
        let Some(pool) = &self.pool else {
            return;
        };
        let _ = sqlx::query("UPDATE jobs SET error_code=$2, updated_at=now() WHERE id=$1")
            .bind(id)
            .bind(code)
            .execute(pool)
            .await;
    }

    /// 孤儿任务回收:扫描所有 `status IN ('queued','running')` 且
    /// `heartbeat_at IS NULL OR heartbeat_at < now() - 5min` 的行,
    /// 把它们置为 error,error='orphaned by server restart'。
    ///
    /// **适用场景**:服务器在 job 跑到一半时挂掉 / SIGKILL / OOM,
    /// DB 里留下了 status='running' 但内存里已经没有对应 JobRegistry
    /// 条目的"僵尸行"。它们会让 `/api/jobs?status=running` 等查询
    /// 永远返回这个 ID,前端看到的是"一直卡在 running"。
    ///
    /// **触发时机**:每次 server 启动后,migrate() 之后,listen bind
    /// 之前(确保孤儿不占任何 in-flight 槽位)。
    ///
    /// **DB 不可用时**:返回 None,调用方降级为日志警告 + 继续启动
    /// (用户接受"重启后看不到这些状态" vs "服务起不来")。
    pub async fn reap_orphans(&self, stale_secs: i64) -> Option<u64> {
        let pool = self.pool.as_ref()?;
        // 用 to_timestamp(seconds) 而非 epoch 转换,PG 端 `now() - interval`
        // 自动按 timestamptz 算;老行(迁移前写入) heartbeat_at IS NULL,
        // `IS NULL OR <` 表达,直接命中。
        let res = sqlx::query(
            "UPDATE jobs SET status='error', \
             error='orphaned by server restart', \
             error_code='orphaned', \
             finished_ms=$2, \
             updated_at=now() \
             WHERE status IN ('queued','running') \
               AND (heartbeat_at IS NULL OR heartbeat_at < now() - make_interval(secs => $1))",
        )
        .bind(stale_secs as f64)
        .bind(now_ms_u64() as i64)
        .execute(pool)
        .await;
        match res {
            Ok(r) => Some(r.rows_affected()),
            Err(e) => {
                eprintln!("[persist] reap_orphans failed: {e}");
                None
            }
        }
    }

    /// 单帧写(现在 run_job 走 add_frames_batch 批量路径;保留单帧 API
    /// 供 image 任务 / 调试工具复用)。
    #[allow(dead_code)]
    pub async fn add_frame(&self, job_id: &str, f: &FrameResult) {
        let Some(pool) = &self.pool else {
            return;
        };
        // 帧行 upsert。
        let _ = sqlx::query(
            "INSERT INTO frames (job_id, idx, timestamp_ms, annotated_key, original_key) VALUES ($1,$2,$3,$4,$5)
             ON CONFLICT (job_id, idx) DO NOTHING"
        )
        .bind(job_id).bind(f.index as i64).bind(f.timestamp_ms as i64)
        .bind(&f.annotated_key).bind(&f.original_key)
        .execute(pool).await;
        for (i, face) in f.faces.iter().enumerate() {
            let _ = sqlx::query(
                "INSERT INTO faces (job_id, frame_idx, face_idx, key, x, y, w, h, score)
                 VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) ON CONFLICT DO NOTHING",
            )
            .bind(job_id)
            .bind(f.index as i64)
            .bind(i as i32)
            .bind(&face.key)
            .bind(face.x as i32)
            .bind(face.y as i32)
            .bind(face.w as i32)
            .bind(face.h as i32)
            .bind(face.score)
            .execute(pool)
            .await;
        }
    }

    /// 批量写帧(一次往返写完一个 job 积攒的多帧 + 其全部 face),
    /// 把写放大从 每帧 1+N 次网络往返 降为 每批 2 条语句。
    /// 内部用 UNNEST 做集合绑定;任一批失败仅打日志(persist 是尽力而为)。
    pub async fn add_frames_batch(&self, job_id: &str, frames: &[FrameResult]) {
        if frames.is_empty() {
            return;
        }
        let Some(pool) = &self.pool else {
            return;
        };
        // 1) frames 表
        let n = frames.len();
        let mut idx = Vec::with_capacity(n);
        let mut ts = Vec::with_capacity(n);
        let mut ak: Vec<Option<String>> = Vec::with_capacity(n);
        let mut ok: Vec<Option<String>> = Vec::with_capacity(n);
        for f in frames {
            idx.push(f.index as i64);
            ts.push(f.timestamp_ms as i64);
            ak.push(f.annotated_key.clone());
            ok.push(f.original_key.clone());
        }
        if let Err(e) = sqlx::query(
            "INSERT INTO frames (job_id, idx, timestamp_ms, annotated_key, original_key)
             SELECT $1, * FROM UNNEST($2::bigint[], $3::bigint[], $4::text[], $5::text[])
             ON CONFLICT (job_id, idx) DO NOTHING",
        )
        .bind(job_id)
        .bind(&idx)
        .bind(&ts)
        .bind(&ak)
        .bind(&ok)
        .execute(pool)
        .await
        {
            tracing::warn!("[persist] add_frames_batch frames failed: {e}");
            return;
        }
        // 2) faces 表(展平全部帧的 face)
        let total_faces: usize = frames.iter().map(|f| f.faces.len()).sum();
        if total_faces == 0 {
            return;
        }
        let mut fidx = Vec::with_capacity(total_faces);
        let mut face_idx = Vec::with_capacity(total_faces);
        let mut key = Vec::with_capacity(total_faces);
        let mut x = Vec::with_capacity(total_faces);
        let mut y = Vec::with_capacity(total_faces);
        let mut w = Vec::with_capacity(total_faces);
        let mut h = Vec::with_capacity(total_faces);
        let mut score = Vec::with_capacity(total_faces);
        for f in frames {
            for (i, face) in f.faces.iter().enumerate() {
                fidx.push(f.index as i64);
                face_idx.push(i as i32);
                key.push(face.key.clone());
                x.push(face.x as i32);
                y.push(face.y as i32);
                w.push(face.w as i32);
                h.push(face.h as i32);
                score.push(face.score);
            }
        }
        if let Err(e) = sqlx::query(
            "INSERT INTO faces (job_id, frame_idx, face_idx, key, x, y, w, h, score)
             SELECT $1, * FROM UNNEST($2::bigint[], $3::int[], $4::text[], $5::int[], $6::int[], $7::int[], $8::int[], $9::real[])
             ON CONFLICT DO NOTHING"
        )
        .bind(job_id).bind(&fidx).bind(&face_idx).bind(&key)
        .bind(&x).bind(&y).bind(&w).bind(&h).bind(&score)
        .execute(pool).await
        {
            tracing::warn!("[persist] add_frames_batch faces failed: {e}");
        }
    }

    pub async fn update_job_stats(&self, id: &str, s: &JobStats) {
        let Some(pool) = &self.pool else {
            return;
        };
        let _ = sqlx::query(
            "UPDATE jobs SET frames_processed=$2, frames_with_face=$3, total_detections=$4 WHERE id=$1"
        )
        .bind(id)
        .bind(s.frames_processed as i64)
        .bind(s.frames_with_face as i64)
        .bind(s.total_detections as i64)
        .execute(pool).await;
    }

    /// 删除单个 job(frames / faces 通过 FK ON DELETE CASCADE 自动删)。
    /// 返回是否真删了(true=行被删,false=id 不存在或 DB 未启用)。
    pub async fn delete_job(&self, id: &str) -> bool {
        let Some(pool) = &self.pool else {
            return false;
        };
        match sqlx::query("DELETE FROM jobs WHERE id=$1")
            .bind(id)
            .execute(pool)
            .await
        {
            Ok(r) => r.rows_affected() > 0,
            Err(e) => {
                tracing::warn!("[persist] delete_job({id}) failed: {e}");
                false
            }
        }
    }

    /// 批量删除:`DELETE FROM jobs WHERE id = ANY($1)`。返回被删的行数。
    pub async fn delete_jobs(&self, ids: &[String]) -> u64 {
        if ids.is_empty() {
            return 0;
        }
        let Some(pool) = &self.pool else {
            return 0;
        };
        match sqlx::query("DELETE FROM jobs WHERE id = ANY($1)")
            .bind(ids)
            .execute(pool)
            .await
        {
            Ok(r) => r.rows_affected(),
            Err(e) => {
                tracing::warn!("[persist] delete_jobs({ids:?}) failed: {e}");
                0
            }
        }
    }

    /// 从 PG 恢复任务(用于 server 重启)。返回每个 job 的所有 frame + face。
    #[allow(dead_code)] // 保留:server 重启后的 job 恢复路径
    pub async fn list_jobs(&self) -> Vec<serde_json::Value> {
        let Some(pool) = &self.pool else {
            return vec![];
        };
        let rows = sqlx::query(
            "SELECT id, kind, display_name, status, created_ms, finished_ms, frames_processed, frames_with_face, total_detections, original_key, error, algo FROM jobs ORDER BY created_ms DESC"
        )
        .fetch_all(pool).await.unwrap_or_default();
        rows.into_iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.get::<String, _>("id"),
                    "kind": r.get::<String, _>("kind"),
                    "display_name": r.get::<String, _>("display_name"),
                    "status": r.get::<String, _>("status"),
                    "created_ms": r.get::<i64, _>("created_ms"),
                    "finished_ms": r.get::<Option<i64>, _>("finished_ms"),
                    "stats": {
                        "frames_processed": r.get::<i64, _>("frames_processed"),
                        "frames_with_face": r.get::<i64, _>("frames_with_face"),
                        "total_detections": r.get::<i64, _>("total_detections"),
                        "elapsed_ms": 0u64,
                    },
                    "algo": r.get::<Option<String>, _>("algo"),
                    "face_count": r.get::<i64, _>("total_detections"), // 近似
                    "frame_count": 0,
                    "original_key": r.get::<Option<String>, _>("original_key"),
                    "error": r.get::<Option<String>, _>("error"),
                })
            })
            .collect()
    }

    #[allow(dead_code)] // 保留:job 详情的 DB 直读路径(内存 miss 时兜底)
    pub async fn list_frames(&self, job_id: &str) -> Vec<FrameResult> {
        let Some(pool) = &self.pool else {
            return vec![];
        };
        let rows = sqlx::query(
            "SELECT idx, timestamp_ms, annotated_key, original_key FROM frames WHERE job_id=$1 ORDER BY idx ASC"
        )
        .bind(job_id).fetch_all(pool).await.unwrap_or_default();
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let idx: i64 = r.get("idx");
            let faces = self.list_faces(job_id, idx).await;
            out.push(FrameResult {
                index: idx as u64,
                timestamp_ms: r.get::<i64, _>("timestamp_ms") as u64,
                annotated_key: r.get("annotated_key"),
                original_key: r.get("original_key"),
                faces,
            });
        }
        out
    }

    #[allow(dead_code)]
    async fn list_faces(&self, job_id: &str, frame_idx: i64) -> Vec<FaceEntry> {
        let Some(pool) = &self.pool else {
            return vec![];
        };
        let rows = sqlx::query(
            "SELECT key, x, y, w, h, score FROM faces WHERE job_id=$1 AND frame_idx=$2 ORDER BY face_idx ASC"
        )
        .bind(job_id).bind(frame_idx).fetch_all(pool).await.unwrap_or_default();
        rows.into_iter()
            .map(|r| FaceEntry {
                key: r.get("key"),
                x: r.get::<i32, _>("x") as usize,
                y: r.get::<i32, _>("y") as usize,
                w: r.get::<i32, _>("w") as usize,
                h: r.get::<i32, _>("h") as usize,
                score: r.get("score"),
            })
            .collect()
    }
}

fn status_to_str(s: JobStatus) -> &'static str {
    match s {
        JobStatus::Queued => "queued",
        JobStatus::Running => "running",
        JobStatus::Done => "done",
        JobStatus::Cancelled => "cancelled",
        JobStatus::Error => "error",
    }
}

/// 迁移切词器的词法状态。
enum Lex {
    Normal,
    LineComment,
    /// `/* */` 块注释;PG 允许嵌套,u32 记录嵌套深度。
    Block(u32),
    /// `'...'` 字符串;bool=true 表示 E'...' 风格,反斜杠转义下一字符。
    Single(bool),
    /// `"..."` 引用标识符。
    Double,
    /// `$tag$ ... $tag$` 美元引用(函数体常见),tag 可为空串。
    Dollar(String),
}

/// 按顶层 `;` 把 SQL 脚本切成单条语句。与朴素的 `sql.split(';')` 不同,
/// 切词器跟踪 PG 的词法结构,注释 / 字面量里的 `;` 不会误切:
/// - `--` 行注释、`/* */` 嵌套块注释,
/// - `'...'` 字符串(`''` 双写、E'...' 的 `\` 转义),
/// - `"..."` 标识符(`""` 双写),
/// - `$tag$...$tag$` 美元引用。
fn split_sql_statements(sql: &str) -> Vec<String> {
    let chars: Vec<char> = sql.chars().collect();
    let n = chars.len();
    let mut stmts = Vec::new();
    let mut cur = String::new();
    let mut state = Lex::Normal;
    let mut i = 0usize;
    while i < n {
        let c = chars[i];
        let next = if i + 1 < n { Some(chars[i + 1]) } else { None };
        match state {
            Lex::Normal => {
                if c == '-' && next == Some('-') {
                    cur.push_str("--");
                    i += 2;
                    state = Lex::LineComment;
                } else if c == '/' && next == Some('*') {
                    cur.push_str("/*");
                    i += 2;
                    state = Lex::Block(1);
                } else if c == '\'' {
                    // E'...' / e'...' 前缀:前一个字符是 E 且再往前不是
                    // 标识符字符,才按转义字符串处理。
                    let prev_is_e =
                        matches!(chars.get(i.wrapping_sub(1)), Some(&e) if e == 'E' || e == 'e');
                    let before_is_ident = matches!(
                        chars.get(i.wrapping_sub(2)),
                        Some(&p) if p.is_ascii_alphanumeric() || p == '_'
                    );
                    let escaped = prev_is_e && !before_is_ident;
                    cur.push(c);
                    i += 1;
                    state = Lex::Single(escaped);
                } else if c == '"' {
                    cur.push(c);
                    i += 1;
                    state = Lex::Double;
                } else if c == '$' {
                    if let Some((tag, end)) = dollar_tag_at(&chars, i) {
                        for ch in &chars[i..end] {
                            cur.push(*ch);
                        }
                        i = end;
                        state = Lex::Dollar(tag);
                    } else {
                        cur.push(c);
                        i += 1;
                    }
                } else if c == ';' {
                    if !cur.trim().is_empty() {
                        stmts.push(std::mem::take(&mut cur));
                    }
                    i += 1;
                } else {
                    cur.push(c);
                    i += 1;
                }
            }
            Lex::LineComment => {
                cur.push(c);
                i += 1;
                if c == '\n' {
                    state = Lex::Normal;
                }
            }
            Lex::Block(depth) => {
                if c == '/' && next == Some('*') {
                    cur.push_str("/*");
                    i += 2;
                    state = Lex::Block(depth + 1);
                } else if c == '*' && next == Some('/') {
                    cur.push_str("*/");
                    i += 2;
                    state = if depth <= 1 {
                        Lex::Normal
                    } else {
                        Lex::Block(depth - 1)
                    };
                } else {
                    cur.push(c);
                    i += 1;
                }
            }
            Lex::Single(escaped) => {
                if escaped && c == '\\' {
                    if let Some(nc) = next {
                        cur.push(c);
                        cur.push(nc);
                        i += 2;
                    } else {
                        cur.push(c);
                        i += 1;
                    }
                } else if c == '\'' {
                    if next == Some('\'') {
                        cur.push_str("''");
                        i += 2;
                    } else {
                        cur.push(c);
                        i += 1;
                        state = Lex::Normal;
                    }
                } else {
                    cur.push(c);
                    i += 1;
                }
            }
            Lex::Double => {
                if c == '"' && next == Some('"') {
                    cur.push_str("\"\"");
                    i += 2;
                } else {
                    cur.push(c);
                    i += 1;
                    if c == '"' {
                        state = Lex::Normal;
                    }
                }
            }
            Lex::Dollar(ref tag) => {
                if let Some((t, end)) = dollar_tag_at(&chars, i) {
                    if &t == tag {
                        for ch in &chars[i..end] {
                            cur.push(*ch);
                        }
                        i = end;
                        state = Lex::Normal;
                        continue;
                    }
                }
                cur.push(c);
                i += 1;
            }
        }
    }
    if !cur.trim().is_empty() {
        stmts.push(cur);
    }
    stmts
}

/// 若 `start` 处是合法的美元引用开标签 `$tag$`,返回 (tag, 闭 `$` 后的下标)。
/// tag 可空;非空时必须遵循未引用标识符规则(字母/下划线开头)。
fn dollar_tag_at(chars: &[char], start: usize) -> Option<(String, usize)> {
    if chars[start] != '$' {
        return None;
    }
    let mut j = start + 1;
    if j < chars.len() && !(chars[j].is_ascii_alphabetic() || chars[j] == '_') {
        // 空 tag(紧接着闭 $)合法,否则首字符非法(如 $1 参数)。
        if j < chars.len() && chars[j] == '$' {
            return Some((String::new(), j + 1));
        }
        return None;
    }
    while j < chars.len() && (chars[j].is_ascii_alphanumeric() || chars[j] == '_') {
        j += 1;
    }
    if j < chars.len() && chars[j] == '$' {
        Some((chars[start + 1..j].iter().collect(), j + 1))
    } else {
        None
    }
}

pub fn now_ms_u64() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::split_sql_statements;

    #[test]
    fn splits_on_top_level_semicolons() {
        let stmts = split_sql_statements("CREATE TABLE a (x INT); CREATE TABLE b (y INT);");
        assert_eq!(stmts.len(), 2);
        assert!(stmts[0].contains("CREATE TABLE a"));
        assert!(stmts[1].contains("CREATE TABLE b"));
    }

    #[test]
    fn semicolon_in_string_literal_does_not_split() {
        let stmts = split_sql_statements("INSERT INTO t VALUES ('a;b'); SELECT 1;");
        assert_eq!(stmts.len(), 2);
        assert!(stmts[0].contains("'a;b'"));
    }

    #[test]
    fn doubled_quote_keeps_semicolon_inside() {
        let stmts = split_sql_statements("VALUES ('it''s; ok'); SELECT 1;");
        assert_eq!(stmts.len(), 2);
        assert!(stmts[0].contains("'it''s; ok'"));
    }

    #[test]
    fn e_string_backslash_does_not_terminate() {
        let stmts = split_sql_statements(r"VALUES (E'x\';y'); SELECT 1;");
        assert_eq!(stmts.len(), 2);
        assert!(stmts[0].contains(r"E'x\';y'"));
    }

    #[test]
    fn line_comment_semicolon_is_ignored() {
        let stmts = split_sql_statements("-- x; y\nSELECT 1; SELECT 2;");
        assert_eq!(stmts.len(), 2);
        assert!(stmts[0].trim_start().starts_with("-- x; y"));
    }

    #[test]
    fn nested_block_comments_have_one_statement() {
        let stmts = split_sql_statements("/* a /* b */ c; */ SELECT 1;");
        assert_eq!(stmts.len(), 1);
        assert!(stmts[0].starts_with("/* a /* b */ c; */"));
    }

    #[test]
    fn dollar_quoted_body_keeps_semicolons() {
        let sql = "CREATE FUNCTION f() RETURNS int AS $$ BEGIN\nRETURN 1;\nEND $$ LANGUAGE plpgsql; SELECT 2;";
        let stmts = split_sql_statements(sql);
        assert_eq!(stmts.len(), 2);
        assert!(stmts[0].contains("RETURN 1;"));
    }

    #[test]
    fn tagged_dollar_quote_matches_only_same_tag() {
        let sql = "DO $body$ BEGIN PERFORM ';'; END $body$; SELECT 1;";
        let stmts = split_sql_statements(sql);
        assert_eq!(stmts.len(), 2);
        assert!(stmts[0].contains("$body$"));
    }

    #[test]
    fn bundled_migrations_split_to_expected_statements() {
        let init = split_sql_statements(include_str!("../../migrations/0001_init.sql"));
        assert_eq!(init.len(), 7);
        let telemetry = split_sql_statements(include_str!("../../migrations/0002_telemetry.sql"));
        assert_eq!(telemetry.len(), 1);
        let health = split_sql_statements(include_str!(
            "../../migrations/0003_jobs_health_columns.sql"
        ));
        // 5 列 ALTER + 5 索引 CREATE = 10 top-level statements
        // (started_at / heartbeat_at / updated_at / archived / error_code;
        //  BRIN × 2 + status B-tree + archived 组合 + error_code B-tree)
        assert!(
            health.len() >= 10,
            "0003 should split into ≥10 stmts, got {}",
            health.len()
        );
        let heartbeat =
            split_sql_statements(include_str!("../../migrations/0004_jobs_heartbeat.sql"));
        // ALTER + UPDATE + CREATE INDEX = 3 top-level statements
        assert_eq!(
            heartbeat.len(),
            3,
            "0004 should split into 3 stmts (ALTER + UPDATE + CREATE INDEX), got {}",
            heartbeat.len()
        );
    }
}
