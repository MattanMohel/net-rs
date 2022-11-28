use std::f32::consts::E;

/// Enumerated network activation function
#[derive(Clone, Copy)]
pub enum Step {
    Sig,
    Lin
}

use Step::*;

impl Step {
    pub fn value(&self, x: f32) -> f32 {
        match self {
            Lin => x,
            Sig => 1. / (1. + E.powf(-x))
        }
    }

    pub fn deriv(&self, x: f32) -> f32 {
        match self {
            Lin => 1.,
            Sig => 1. / (2. + E.powf(x) + E.powf(-x))
        }
    }
}