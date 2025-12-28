/// Represents a 3D vector (position or velocity).
#[derive(Debug, Clone, Copy, Default)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<boxcars::Vector3f> for Vector3 {
    fn from(v: boxcars::Vector3f) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

/// Represents a quaternion rotation.
///
/// Quaternions are preferred over Euler angles for ML because:
/// - They are continuous (no sudden jumps at angle boundaries)
/// - No gimbal lock issues
/// - Neural networks handle them well
#[derive(Debug, Clone, Copy, Default)]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl From<boxcars::Quaternion> for Quaternion {
    fn from(q: boxcars::Quaternion) -> Self {
        Self {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        }
    }
}
