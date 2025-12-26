use chrono::{DateTime, Utc};
use uuid::Uuid;

/// ML model metadata stored in the database.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Model {
    pub id: Uuid,
    pub name: String,
    pub version: i32,
    pub checkpoint_path: String,
    pub training_config: Option<serde_json::Value>,
    pub metrics: Option<serde_json::Value>,
    pub trained_at: DateTime<Utc>,
}
