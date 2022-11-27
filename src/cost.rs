
/// Enumerated network cost function
#[derive(Clone, Copy)]
pub enum Cost {
    Quad,
}

use Cost::*;

impl Cost {
    pub fn value(&self, diff: f32) -> f32 {
        match self {
            Quad => diff.powi(2)
        }
    }

    pub fn deriv(&self,  diff: f32) -> f32 {
        match self {
            Quad => 2. * diff
        }
    }
}