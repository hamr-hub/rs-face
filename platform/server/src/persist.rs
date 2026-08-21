//! PostgreSQL 持久化层:job/frame/face 三张表。
//!
//! - DB 不可用时 JobRegistry 降级为纯内存(不报错),保留可用性;
//! - 所有写入走 `persist_*` 函数,无返回值(void 风格)。

use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;

use crate::jobs::{FaceEntry, FrameResult, JobKind, JobStats, JobStatus};

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
                eprintln!("[persist] connected to PostgreSQL");
                Self { pool: Some(pool) }
            }
            Err(e) => {
                eprintln!("[persist] PG connect failed: {e} — running in memory-only mode");
                Self { pool: None }
            }
        }
    }

    pub async fn migrate(&self) {
        let Some(pool) = &self.pool else {
            return;
        };
        let sql = include_str!("../../migrations/0001_init.sql");
        // sqlx::query 不支持多语句,按 `;` 切分逐条执行。
        for stmt in sql.split(';') {
            let s = stmt.trim();
            if s.is_empty() {
                continue;
            }
            if let Err(e) = sqlx::query(s).execute(pool).await {
                eprintln!("[persist] migration statement failed: {e}\n>> {s}");
            }
        }
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
            eprintln!("[persist] add_frames_batch frames failed: {e}");
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
            eprintln!("[persist] add_frames_batch faces failed: {e}");
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
                eprintln!("[persist] delete_job({id}) failed: {e}");
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
                eprintln!("[persist] delete_jobs({ids:?}) failed: {e}");
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

pub fn now_ms_u64() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
