use std::ops::Range;
use super::step::Step;
use super::cost::Cost;
use crate::linalg::LinAlg;
use crate::linalg::LinAlgGen;
use crate::linalg::LinAlgMul;
use crate::linalg::Vector;
use crate::linalg::Matrix;
use crate::array::Array;
use crate::array::IndexType::*;

const LEARN_RATE: f32 = 0.01;
const BATCH_SIZE: usize = 64;

pub struct Data {
    // layer sizes
    pub form: Vec<usize>,
    // size of batch sampling
    pub batch_size: usize,
    // learning coefficient
    pub learn_rate: f32,
    // activation function
    pub step: Step,
    // cost function
    pub cost: Cost,
}

impl Data {
    pub fn new(form: &[usize]) -> Self {
        Self { 
            form: form.to_vec(),
            batch_size: BATCH_SIZE, 
            learn_rate: LEARN_RATE, 
            step: Step::Sig, 
            cost: Cost::Quad, 
        }
    }

    fn zero_array<M, R, F>(&self, range: R, func: F) -> Array<M>
    where
        M: LinAlg,
        R: Fn(usize) -> Range<usize>,
        F: Fn(usize, &[usize]) -> M::Dim,   
    {
        let buf = range(self.form.len())
            .map(|i| M::from_zeros(func(i, &self.form)))
            .collect();

        Array::from_buf(buf)
    }

    fn rand_array<M, R, F>(&self, range: R, func: F) -> Array<M> 
    where
        M: LinAlg,
        R: Fn(usize) -> Range<usize>,
        F: Fn(usize, &[usize]) -> M::Dim,   
    {
        let buf = range(self.form.len())
            .map(|i| M::rand(func(i, &self.form)))
            .collect();

        Array::from_buf(buf)
    }
}

pub struct Net<const L: usize> {
    // hyper parameters
    data: Data,
    // weight matrices layers
    weights: Array<Matrix>,
    // bias matrices layers
    biases: Array<Vector>,

    /////////////////////
    /// Training Data ///
    /////////////////////

    // activations buffer
    acts:      Array<Vector>,
    // layer sums buffer
    sums:      Array<Vector>,
    // layer error buffer
    err:       Array<Vector>,
    // layer error accumulation buffer
    acc_err:   Array<Vector>,
    // weight error buffer
    w_err:     Array<Matrix>,
    // weight error accumulation buffer
    acc_w_err: Array<Matrix>,
}

impl<const L: usize> Net<L> {
    pub fn new(form: [usize; L]) -> Self {
        if form.len() <= 2 {
            panic!()
        }

        let data = Data::new(&form);

        Self {
            acts:      data.zero_array(|l| 0..l,   |i, f| f[i]),
            sums:      data.zero_array(|l| 1..l,   |i, f| f[i]),
            err:       data.zero_array(|l| 1..l,   |i, f| f[i]),
            acc_err:   data.zero_array(|l| 1..l,   |i, f| f[i]),
            biases:    data.zero_array(|l| 1..l,   |i, f| f[i]),          
            weights:   data.rand_array(|l| 0..l-1, |i, f| ( f[i+1], f[i] )),
            w_err:     data.zero_array(|l| 0..l-1, |i, f| ( f[i+1], f[i] )),
            acc_w_err: data.zero_array(|l| 0..l-1, |i, f| ( f[i+1], f[i] )),
            data,
        }
    }

    /// Clears propagation buffers
    pub fn clear_propagation_data(&mut self) {
        self.acts.fill(0_f32);
        self.sums.fill(0_f32);
        self.err.fill(0_f32);
        self.w_err.fill(0_f32);
    }

    /// Clears accumulation buffers
    pub fn clear_accumulation_data(&mut self) {
        self.acc_w_err.fill(0_f32);
        self.acc_err.fill(0_f32);
    }

    pub fn len(&self) -> usize {
        self.data.form.len() - 1
    }

    /// Applies activation function to one number
    fn step(&self, n: f32) -> f32 {
        self.data.step.value(n)
    }

    /// Applies activation derivative to one number
    fn d_step(&self, n: f32) -> f32 {
        self.data.step.deriv(n)
    }

    /// Applies cost function to one number
    fn cost(&self, diff: f32) -> f32 {
        self.data.cost.value(diff)
    }

    /// Applies cost derivative to one number
    fn d_cost(&self, diff: f32) -> f32 {
        self.data.cost.deriv(diff)
    }

    pub fn apply_gradient(&mut self, samples: usize) {
        let learn_rate = self.data.learn_rate / samples as f32;

        // apply stochastic error gradient 
        for j in 0..L-1 {
            self.biases[j].add_eq(&self.acc_err[j].scale(learn_rate));
            self.weights[j].add_eq(&self.acc_w_err[j].scale(learn_rate));
        }
    }

    pub fn forward_prop(&mut self, input: &Vector) -> &Vector {
        if input.row() != self.data.form[0] {
            panic!()
        }

        self.acts[0] = input.clone();

        for lr in 0..L-1 {            
            self.weights[lr].mul_to(&self.acts[lr], &mut self.sums[lr]);
            self.sums[lr].add_eq(&self.biases[lr]);
            
            self.acts[lr+1] = self.sums[lr].map(|n| self.step(n));
        }

        &self.acts[Back(0)]
    }

    pub fn back_prop(&mut self, input: &Vector, exp: &Vector) {          
        // clear prior training data
        self.clear_propagation_data(); 

        // propogate input and store actual output
        self.forward_prop(input);

        // error_L = cost' ( y - a_L ) . sum_L
        // self.err[Back(0)] = exp
        //     .sub(&self.acts[Back(0)])
        //     .map_eq(|n| self.d_cost(n))
        //     .dot(&self.sums[Back(0)].map(|n| self.d_step(n)));

        self.sums[Back(0)]
            .map(|n| self.d_step(n))
            .to_diagonal()
            .mul_to(
                &exp.sub(&self.acts[Back(0)]).map(|n| self.d_step(n)),
                &mut self.err[Back(0)]
            );

        for lr in 0..L-1 {
            // weight_l = error_l x activations_l-1 ^ T
            self.err[Back(lr)]
                .mul_t2_to(&self.acts[Back(1+lr)], &mut self.w_err[Back(lr)]);
                
            // self.w_err[Back(lr)].mul_t2_to(&self.err[Back(lr)], &self.acts[Back(1+lr)]);

            if lr == L-2 {
                break
            }

            self.sums[Back(lr+1)]
                .map(|n| self.d_step(n))
                .to_diagonal()
                .mul_t2::<_, Matrix>(&self.weights[Back(lr)])
                .mul_to(&self.err[Back(lr)].clone(), &mut self.err[Back(lr+1)]);

            // let d_sum = self.sums[Back(lr+1)].map(|n| self.d_step(n));
            // let split_index = self.err.len() - (lr + 1);
            // let (l1, l2) = self.err.buf.split_at_mut(split_index);

            // l1[l1.len()-1]
            //     .mul_t1_to(&self.weights[Back(lr)], &l2[0])
            //     .dot_eq(&d_sum);
        }

        // self.err.buf.reverse();
        // self.w_err.buf.reverse();
    }

    pub fn train(&mut self, inputs: &[Vector], expected: &[Vector], take: usize) {
        if inputs.len() != expected.len() {
            panic!()
        }

        self.clear_accumulation_data();

        for (i, (input, expected)) in inputs.iter().take(take).zip(expected.iter()).enumerate() {
            self.back_prop(input, expected);
            
            // accumulate errors
            for j in 0..self.err.len() {
                self.acc_err[j].add_eq(&self.err[j]);
                self.acc_w_err[j].add_eq(&self.w_err[j]);
            }

            // apply gradient
            if i % self.data.batch_size == 0 {
                self.apply_gradient(self.data.batch_size);
                self.clear_accumulation_data();
            }
        }
        
        let remainder = inputs.len() % self.data.batch_size;
        self.apply_gradient(remainder);
    }
}