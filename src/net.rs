use super::act::Act;
use super::cost::Cost;
use super::mat::*;
use super::num::Num;
use super::num::N;

use std::fmt;

const LEARN_RATE: f64 = 0.01;
const BATCH_SIZE: usize = 64;

pub struct Net<const L: usize> {
    pub form: [usize; L],

    pub weights: Vec<Matrix<N>>,
    pub biases:  Vec<Matrix<N>>,

    pub batch_size: usize,
    pub learn_rate: N,

    pub cost: Cost,
    pub activation: Act
}

impl<const L: usize> Net<L> {
    pub fn new(form: [usize; L]) -> Self {
        if form.len() <= 2 {
            panic!()
        }

        Self { 
            cost: Cost::Quad,
            activation:  Act::Sig,

            batch_size: BATCH_SIZE,
            learn_rate: LEARN_RATE,

            biases:  Self::new_biases(&form),
            weights: Self::new_weights(&form),
            form
        }
    }

    pub fn new_weights(form: &[usize; L]) -> Vec<Matrix<N>> {
        (0..form.len() - 1)
            .map(|i| {
                let l_0 = form[i];
                let l_1 = form[i+1];
                Matrix::random((l_1, l_0), -N::one(), N::one()).scale(l_0 as N)
            })
            .collect()
    }

    pub fn new_biases(form: &[usize; L]) -> Vec<Matrix<N>> {
        (0..form.len() - 1)
            .map(|i| Matrix::zeros((form[i+1], 1)))
            .collect()
    }

    pub fn layers(&self) -> usize {
        L - 1
    }

    fn activate(&self, n: N) -> N {
        self.activation.act(n)
    }

    fn d_activate(&self, n: N) -> N {
        self.activation.act(n)
    }

    fn d_cost(&self, n: N, exp: N) -> N {
        self.cost.d_cost(n, exp)
    }

    pub fn forward_prop(&self, input: &Matrix<N>) -> Matrix<N> {
        if input.row() != self.form[0] {
            panic!()
        }

        self.weights
            .iter()
            .zip(self.biases.iter())
            .fold(input.clone(), |acc, (w, b)| {
                w
                    .mul(&acc)
                    .add(&b)
                    .map(|n| self.activation.act(n))
            })
    }

    pub fn train(&mut self, input: &Matrix<N>, expected: &Matrix<N>) {
        let mut sums = Vec::with_capacity(self.layers());
        let mut activations = Vec::with_capacity(self.layers());
        let mut errors = Vec::with_capacity(self.layers());

        activations.push(input.clone());

        for i in 0..self.layers() {
            let sum_l = self.weights[i].mul(&activations[i]).add(&self.biases[i]);
            let activation_l = sum_l.map(|n| self.activate(n));

            sums.push(sum_l);
            activations.push(activation_l);
        }

        // TODO: generalize error method with enum
        let error = 
            sums[self.layers()-1]
                .map(|n| self.d_activate(n))
                .diagonal()
                .mul(&expected.sub(&activations[self.layers()]).scale(2.0));

        errors.push(error);

        let mut weight_errors = Vec::with_capacity(self.layers());

        for i in 0..self.layers() {
            let l = self.layers() - i;

            let weight_error_l = errors[i].mul(&activations[l-1].transpose());

            weight_errors.push(weight_error_l);

            if l == 1 {
                break
            }

            let error_l = 
                sums[l-2]
                    .map(|n| self.d_activate(n))
                    .diagonal()
                    .mul(&self.weights[l-2])
                    .mul(&errors[i]);

            errors.push(error_l);
        }

        for i in 0..self.layers() {
            self.weights[i].sub_eq(&weight_errors[i]);
            self.biases[i].sub_eq(&errors[i]);
        }
    }
}