use super::num::N;
use super::num_fn::NumeralFn;
use std::f32::consts::E;

/// Enumerated network activation function
#[derive(Clone, Copy)]
pub enum Activation {
    Sig,
    Lin
}

use Activation::*;

impl NumeralFn<N> for Activation {
    fn value(&self, x: N) -> N {
        match self {
            Lin => x,
            Sig => 1. / (1. + E.powf(-x))
        }
    }

    fn deriv(&self, x: N) -> N {
        match self {
            Lin => 1.,
            Sig => 1. / (2. + E.powf(x) + E.powf(-x))
        }
    }
}