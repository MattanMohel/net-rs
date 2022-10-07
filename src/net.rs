use super::act::Activation;
use super::init::Weight;
use super::cost::Cost;
use super::mat::Mat;
use super::num::N;

use std::fmt;

pub struct NetParams<const L: usize> {
    pub form: [usize; L],
    pub batch_size: usize,
    pub learn_rate: N,
    // functions...
    pub activation: Activation,
    pub weight_init: Weight,
    pub cost: Cost,
}

impl<const L: usize> NetParams<L> {
    pub fn new(form: [usize; L]) -> Self {
        Self {
            form,
            batch_size: 64,
            learn_rate: 0.01,
            activation: Activation::Sig,
            weight_init: Weight::He,
            cost: Cost::Quad
        }
    }

    pub fn new_biases(&self) -> Vec<Mat> {
        (1..self.form.len())
            .map(|i| Mat::zeros((self.form[i], 1)))
            .collect()
    }

    pub fn new_weights(&self) -> Vec<Mat> {
        (1..self.form.len())
            .map(|i| Mat::from_fn((self.form[i], self.form[i-1]), || {
                self.weight_init.init(self.form[i-1], self.form[i])
            }))
            .collect()
    }
}

pub struct Net<const L: usize> {
    pub weights: Vec<Mat>,
    pub biases:  Vec<Mat>,
    params: NetParams<L>
}

impl<const L: usize> Net<L> {
    pub fn new(form: [usize; L]) -> Self {
        let params = NetParams::new(form);

        Self { 
            weights: params.new_weights(),
            biases:  params.new_biases(),
            params
        }
    }

    pub fn forward_prop(&self, input: Mat) -> Mat {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .take(self.params.form.len()-1)
            .fold(input, |acc, (w, b)| w.mul(&acc).unwrap().add(&b).unwrap())
    } 

    pub fn params(&self) -> &NetParams<L> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut NetParams<L> {
        &mut self.params
    }
}

impl<const L: usize> fmt::Display for Net<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Network {{\n\nform: {:?}\n", self.params().form)?;

        for i in 0..self.params.form.len() - 1 {
            write!(f, "\nLayer({})\n\nweight:\n\n{}\n\nbias{}\n", i, self.weights[i], self.biases[i])?;
        }
        
        write!(f, "\n}}")
    }
}