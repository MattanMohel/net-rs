use crate::num::N;


pub enum Cost {
    Quad
}

impl Cost {
    pub fn cost(&self, n: N, expected: N) -> N {
        match self {
            Cost::Quad => (expected - n).powi(2)
        }
    }

    pub fn d_cost(&self, n: N, expected: N) -> N {
        match self {
            Cost::Quad => 2. * (expected - n)
        }
    }
}