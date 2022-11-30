use super::num_fn::NumeralFn;
use super::num::N;

/// Enumerated network cost function
#[derive(Clone, Copy)]
pub enum Cost {
    Quad
}

use Cost::*;

impl NumeralFn<(N, N)> for Cost {
    fn value(&self, (e, x): (N, N)) -> N {
        match self {
            Quad => (e - x).powi(2)
        }
    }

    fn deriv(&self, (e, x): (N, N)) -> N {
        match self {
            Quad => 2. * (e - x)
        }
    }
}