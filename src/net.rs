use super::{
    num::Num,
    mat::Mat,
    act::Act,
    init::Init
};

use std::ops::{
    Mul,
    Add
};

pub struct Net<const L: usize> {
    form: [usize; L],
    biases: Vec<Mat>,
    weights: Vec<Mat>
}

impl<const L: usize> Net<L> {
    pub fn from_arr(form: [usize; L], init: Init) -> Self {
        Self { 
            biases: {
                (1..form.len())
                    .map(|i| Mat::zeros((form[i], 1)))
                    .collect()
            },
            weights: {
                (1..form.len())
                    .map(|i| Mat::from_map((form[i], form[i-1]), |_, _| init.eval(form[i-1], form[i])))
                    .collect()
            },
            form
        }
    }

    pub fn forward_prop(&self, input: Mat) -> Mat {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .take(self.form.len()-1)
            .fold(input, |acc, (w, b)| w.mul(&acc).unwrap().add(&b).unwrap())
    } 

    pub fn form(&self) -> [usize; L] {
        self.form
    }
}