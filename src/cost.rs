use serde_derive::{Serialize, Deserialize};

/// Enumerated network cost function
#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Cost {
    Quad,
}

impl Cost {
    pub fn value(&self, diff: f32) -> f32 {
        match self {
            Cost::Quad => diff.powi(2)
        }
    }

    pub fn deriv(&self,  diff: f32) -> f32 {
        match self {
            Cost::Quad => 2. * diff
        }
    }
}