use super::num::Num;

pub enum Init {
    Rand,
    He,
}

impl Init {
    pub fn eval<N: Num>(&self, l: usize, l1: usize) -> f64 {
        let l = l as f64;
        let l1 = l1 as f64;
        
        let r = rand::random::<f64>() * l.max(l1) as f64 + l.min(l1) as f64;
        
        r * {
            match self {
                Init::Rand => 1.,
                Init::He   => (2. /  l).sqrt()
            }
        }
    }
}