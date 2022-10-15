use crate::mat::MatErr;
use crate::mat::MatErrType;

use super::act::Act;
use super::cost::Cost;
use super::mat::Mat;
use super::num::Num;
use super::num::N;

use std::fmt;

pub struct Net<const L: usize> {
    pub form: [usize; L],

    pub weights: Vec<Mat>,
    pub biases:  Vec<Mat>,

    pub batch_size: usize,
    pub learn_rate: N,

    pub act: Act,
    pub cost: Cost,
}

impl<const L: usize> Net<L> {
    pub fn new(form: [usize; L]) -> Result<Self, ()> {
        if form.len() <= 2 {
            return Err(())
        }

        Ok(Self { 
            form,

            weights: {
                (0..form.len() - 1)
                    .map(|i| Mat::random((form[i + 1], form[i]), N::neg(), N::one()).scaled(form[i] as N))
                    .collect()
            },
            biases:  {
                (0..form.len() - 1)
                    .map(|i| Mat::zeros((form[i + 1], 1)))
                    .collect()
            },

            batch_size: 64,
            learn_rate: 0.01,

            act: Act::Sig,
            cost: Cost::Quad
        })
    }

    pub fn len(&self) -> usize {
        L - 1
    }

    pub fn forward_prop(&self, input: &Mat) -> Result<Mat, ()> {
        if input.rows() != self.form[0] {
            return Err(())
        }

        Ok(self.weights
            .iter()
            .zip(self.biases.iter())
            .fold(input.clone(), |acc, (w, b)| {
                w.mul(&acc).unwrap().add(&b).unwrap().iter().map(|e| self.act.act(e)).collect()
            }))
    }

    pub fn train(&mut self, input: Mat, expected: Mat) -> MatErr {
        let mut acts = vec![input];
        let mut sums = Vec::new();

        for i in 0..self.len() {
            let sum = self.weights[i].mul(&acts[i])?.add(&self.biases[i])?;
            let act = sum.mapped(|e| self.act.act(e));

            sums.push(sum);
            acts.push(act);
        }

        let mut i = 0;
        let d_cost = acts[self.len()].mapped(|n| {
            i += 1;
            self.cost.d_cost(n, expected[(i - 1, 0)])
        });

        let mut err = sums[self.len()].col_diagonal()?.mul(&d_cost)?;
        
        let mut d_ws = Vec::new();
        let mut errs = Vec::new();

        for i in (0..self.len()).rev() {
            // let act = acts[i].transpose().as_row(2)?;
            // let d_w = err.col_diagonal()?.mul(&act)?;

            // d_ws.push(d_w);
            // errs.push(err.clone());

            // if i == 0 {
            //     break
            // } 

            // let d_prev_sum = sums[i-1].mapped(|e| self.act.d_act(e)).col_diagonal()?;
            // let weight_err = self.weights[i].transpose().mul(&err)?;

            // err = d_prev_sum.mul(&weight_err)?;
        }

        for i in (0..self.len()).rev() {
            self.weights[i] = self.weights[i].sub(&d_ws[i])?;
            self.biases[i]  = self.biases[i].sub(&errs[i])?;
        }
        
        Ok(())
    }
}

impl<const L: usize> fmt::Display for Net<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Network {{\n\nform: {:?}\n", self.form)?;

        for i in 0..(self.form.len() - 1) {
            write!(f, "\nLayer({})\n\nweight:\n\n{}\n\nbias{}\n", i, self.weights[i], self.biases[i])?;
        }
        
        write!(f, "\n}}")
    }
}