use std::f32::consts::E;
use serde_derive::{Serialize, Deserialize};

/// Enumerated network activation function
#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    Sig,
    Tanh,
    Lin
}


impl Activation {
    pub fn value(&self, x: f32) -> f32 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::Sig =>  1. / (1. + E.powf(-x)),
            Activation::Lin =>  x
        }
    }

    pub fn deriv(&self, x: f32) -> f32 {
        match self {
            Activation::Tanh => 1. - x.tanh().powi(2),
            Activation::Sig =>  1. / (2. + E.powf(x) + E.powf(-x)),
            Activation::Lin =>  1.
        }
    }
}