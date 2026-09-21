//! rsface-platform library:让集成测试 (`tests/integration.rs`) 能调
//! router / config / s3 / jobs 等模块。bin `rsface-server` 仅做启动用,
//! 全部业务逻辑都在这里。

pub mod api;
pub mod cache;
pub mod config;
pub mod error_codes;
pub mod jobs;
pub mod metrics;
pub mod persist;
pub mod rate_limit;
pub mod retry;
pub mod s3;
pub mod slow_log;
pub mod zip;
