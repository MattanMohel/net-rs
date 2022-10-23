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
    pub fn new(form: [usize; L]) -> MatErr<Self> {
        if form.len() <= 2 {
            return Err(MatErrType::Unimplemented)
        }

        Ok(Self { 
            act:  Act::Sig,
            cost: Cost::Quad,

            batch_size: 64,
            learn_rate: 0.01,

            biases:  Self::new_biases(&form),
            weights: Self::new_weights(&form),
            form
        })
    }

    pub fn new_weights(form: &[usize; L]) -> Vec<Mat> {
        (0..form.len() - 1).map(|i| {
                let l1 = form[i];
                let l2 = form[i+1];
                Mat::random((l2, l1), N::neg(), N::one()).scaled(l1 as N)
            })
            .collect()
    }

    pub fn new_biases(form: &[usize; L]) -> Vec<Mat> {
        (0..form.len() - 1).map(|i| {
                let l2 = form[i+1];
                Mat::zeros((l2, 1))
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        L - 1
    }

    fn act(&self, n: N) -> N {
        self.act.act(n)
    }

    fn d_act(&self, n: N) -> N {
        self.act.act(n)
    }

    fn d_cost(&self, n: N, exp: N) -> N {
        self.cost.d_cost(n, exp)
    }

    pub fn forward_prop(&self, input: &Mat) -> MatErr<Mat> {
        if input.rows() != self.form[0] {
            return Err(MatErrType::Unimplemented)
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
                    .into_mat((w.rows(), 1))
                    .unwrap()
            }))
    }

    pub fn train(&mut self, input: &Mat, exp: &Mat) -> MatErr {
        // list of resultant activations
        let mut acts = vec![input.clone()];
        // list of linear summations
        let mut sums = Vec::new();

        // compute forward propagation 
        for i in 0..self.len() {
            let sum = self.weights[i].mul(&acts[i])?.add(&self.biases[i])?;
            let act = sum.mapped(|e| self.act.act(e));

            sums.push(sum);
            acts.push(act);
        }

        // activation of layer L
        let a = &acts[self.len()];
        // linear summation of layer L
        let z = &sums[self.len() - 1];  
        // matrix of cost derivatives
        let d_c = a.mapped_idx(|r, _| self.d_cost(a[(r, 0)], exp[(r, 0)]));
        // matrix of activation derivatives
        let d_z = z.mapped_idx(|r, _| self.d_act(z[(r, 0)]));

        println!("diag: {}", d_z.col_diagonal()?);
        // calculated error of layer L
        let err = d_z.col_diagonal()?.mul(&d_c)?;

        // list of weight derivatives
        let mut d_ws = Vec::new();
        // list of summation derivatives
        let mut d_zs = vec![d_z];
        // list of layer errors
        let mut errs = vec![err];

        // approximately...
        for i in (0..self.len()).rev() {
            let err_l = &errs[self.len() - i];
            let d_w = self.weights[i].mapped_idx(|r, c| acts[i][(0, c)] * err_l[(r, 0)]);
            
            d_ws.push(d_w);
            
            if i == 1 {
                break
            }
            
            let d_z = sums[i - 1].mapped(|n| self.d_act(n)).col_diagonal()?;
            let w_t = self.weights[i].transpose();
            let err = d_z.mul(&w_t)?.mul(err_l)?;

            d_zs.push(d_z);
            errs.push(err);            
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