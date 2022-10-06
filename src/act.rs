use super::num::Num;
use std::f64::consts;

pub enum Act {
    Sig,
    Lin
}

impl Act {
    pub fn act<N: Num>(self, n: f64) -> f64 {
        match self {
            Act::Lin => n,
            Act::Sig => 1. / (1. + consts::E.powf(n))
        }
    }
}