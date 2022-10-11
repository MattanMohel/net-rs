use super::num::N;
use std::f64::consts::E;

#[derive(Clone, Copy)]
pub enum Act {
    Sig,
    Lin
}

use Act::*;

impl Act {
    pub fn act(&self, n: N) -> N {
        match self {
            Lin => n,
            Sig => 1. / (1. + E.powf(-n))
        }
    }

    pub fn d_act(&self, n: N) -> N {
        match self {
            Lin => 1.,
            Sig => 1. / (2. + E.powf(n) + E.powf(-n))
        }
    }
}