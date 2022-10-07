use super::num::N;

#[derive(Clone, Copy)]
pub enum Weight {
    Rand,
    He,
}

impl Weight {
    pub fn init(&self, l: usize, l1: usize) -> N {
        let l  = l  as f64;
        let l1 = l1 as f64;
        
        let r = rand::random::<N>() * l.max(l1) + l.min(l1);
        
        r * {
            match self {
                Weight::Rand => 1.,
                Weight::He   => (2. /  l).sqrt()
            }
        }
    }
}