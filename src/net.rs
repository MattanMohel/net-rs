use crate::mat::MatCollect;
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
            act: Act::Sig,
            cost: Cost::Quad,

            batch_size: 64,
            learn_rate: 0.01,

            biases: Self::new_biases(&form),
            weights: Self::new_weights(&form),
            form
        })
    }

    pub fn new_weights(form: &[usize; L]) -> Vec<Mat> {
        (0..form.len() - 1)
            .map(|i| {
                let l1 = form[i];
                let l2 = form[i+1];
                Mat::random((l2, l1), N::neg(), N::one()).scaled(l1 as N)
            })
            .collect()
    }

    pub fn new_biases(form: &[usize; L]) -> Vec<Mat> {
        (0..form.len() - 1)
            .map(|i| {
                let l2 = form[i+1];
                Mat::zeros((l2, 1))
            })
            .collect()
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
                w
                    .mul(&acc)
                    .unwrap()
                    .add(&b)
                    .unwrap()
                    .iter()
                    .map(|e| self.act.act(*e))
                    .to_matrix((w.rows(), 1))
                    .unwrap()
            }))
    }

    pub fn train(&mut self, input: Mat, exp: Mat) -> MatErr {
        let mut acts = vec![input];
        let mut sums = Vec::new();

        for i in 0..self.len() {
            let sum = self.weights[i].mul(&acts[i])?.add(&self.biases[i])?;
            let act = sum.mapped(|elem| self.act.act(elem));

            sums.push(sum);
            acts.push(act);
        }

        let gradient = acts[self.len()].iter().zip(exp.iter())
            .map(|(a, e)| self.cost.d_cost(*a, *e))
            .to_matrix(exp.dim())?;

        let mut err = sums[self.len()]
            .hadamard(&gradient)?
            .col_diagonal()?;
        
        let mut d_ws = Vec::new();
        let mut errs: Vec<Mat> = Vec::new();

        for i in (0..self.len()).rev() {
            let l2 = self.form[i];
            let l1 = self.form[i-1];


            // TODO: implement from_cols/from_rows and use here
            for j in 0..l2 {
                acts[i-1].scaled(err[(j, 0)]);
            }

            let d_w = Mat::from_map((self.form[i], self.form[i-1]), |r, c| {
                errs[i][(r, 0)] * acts[i-1][(r, c)]
            });

            let d_b = &errs[i];

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