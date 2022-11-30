use super::num::N;

/// Trait for Network enumerated numeric functions
pub trait NumeralFn<I, O=N> {    
    fn value(&self, x: I) -> O;
    fn deriv(&self, x: I) -> O;
}