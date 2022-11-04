use super::act::Act;
use super::cost::Cost;
use super::mat::*;
use super::num::Num;
use super::num::N;

use std::fmt;

const LEARN_RATE: f64 = 0.01;
const BATCH_SIZE: usize = 64;
const TRAIN_EPOCH: usize = 5;

pub struct Net<const L: usize> {
    pub form: [usize; L],

    pub weights: Vec<Matrix>,
    pub biases:  Vec<Matrix>,

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

    pub fn new_weights(form: &[usize; L]) -> Vec<Matrix> {
        (0..form.len() - 1)
            .map(|i| {
                let l_0 = form[i];
                let l_1 = form[i+1];
                Matrix::random((l_1, l_0), -N::one(), N::one()).scale(l_0 as N)
            })
            .collect()
    }

    pub fn new_biases(form: &[usize; L]) -> Vec<Matrix> {
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

    pub fn forward_prop(&self, input: &Matrix<N>) -> Matrix {
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

    pub fn backward_prop(&mut self, input: Vec<&Matrix>, expected: Vec<&Matrix>) {
        let epoch_steps = (expected.len() as f32 / (self.batch_size) as f32).ceil() as usize;

        for _ in 0..TRAIN_EPOCH {
            // TODO: assert input.len() > 1
            let (mut errors_acc, mut weight_errors_acc) = self.train(input[0], expected[0]);

            for i in 1..epoch_steps {
                let (error, weight_error) = self.train(input[i], expected[i]);
                
                for j in 0..errors_acc.len() {
                    errors_acc[j].add_eq(&error[j]);
                    weight_errors_acc[j].add_eq(&weight_error[j]);
                }

                for j in 0..errors_acc.len() {
                    errors_acc[j].scale(self.learn_rate / self.batch_size as f64);
                    weight_errors_acc[j].scale(self.learn_rate / self.batch_size as f64);
                }

                for (i, l) in (0..L-1).map(|i| (i, L-2-i)) {   
                    self.weights[i].add_eq(&weight_errors_acc[l]);
                    self.biases[i].add_eq(&errors_acc[l]);
                }
            }
        }
    }

    pub fn train(&self, input: &Matrix, expected: &Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut errors = Vec::with_capacity(L-1);
        let mut sums = Vec::with_capacity(L-1);
        let mut activations = Vec::with_capacity(L-1);
        activations.push(input.clone());

        for i in 0..L-1 {
            let sum_l = self.weights[i].mul(&activations[i]).add(&self.biases[i]);
            let activation_l = sum_l.map(|n| self.activate(n));
            activations.push(activation_l);
            sums.push(sum_l);
        }

        // TODO: generalize error method with enum
        let error = 
            sums[L-2]
                .map(|n| self.d_activate(n))
                .diagonal()
                .mul(&expected.sub(&activations[L-1]).scale(2.0));

        errors.push(error);
        let mut weight_errors = Vec::with_capacity(L-1);

        for (i, l) in (0..L-1).map(|i| (i, L-2-i)) {
            let weight_error_l = errors[L-2-l].mul(&activations[l].transpose());
            weight_errors.push(weight_error_l);

            if l == 0 {
                break
            }

            let error_l = 
                sums[l-1]
                    .map(|n| self.d_activate(n))
                    .diagonal()
                    .mul(&self.weights[l].transpose())
                    .mul(&errors[i]);

            errors.push(error_l);
        }

        (errors, weight_errors)
    }
}