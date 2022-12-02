
use std::ops::Range;

use crate::array::Array;

use super::step::Step;
use super::cost::Cost;
use super::linalg::*;

type N = f32;

const LEARN_RATE: N = 0.03;
const BATCH_SIZE: usize = 64;

pub struct Meta {
    // layer sizes
    pub form: Vec<usize>,
    // size of batch sampling
    pub batch_size: usize,
    // learning coefficient
    pub learn_rate: N,
    // step function
    pub step: Step,
    // cost function
    pub cost: Cost,
}

impl Meta {
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

pub struct Network<const L: usize> {
    // hyper parameters
    meta_data: Meta,
    // weight matrices layers
    weights: Array<Matrix>,
    // bias matrices layers
    biases: Array<Matrix>,

    /////////////////////
    /// Training Data ///
    /////////////////////

    // steps buffer
    steps: Array<Matrix>,
    // layer sums buffer
    sums: Array<Matrix>,
    // layer error buffer
    errors: Array<Matrix>,
    // layer error accumulation buffer
    error_acc: Array<Matrix>,
    // weight error buffer
    weight_errors: Array<Matrix>,
    // weight error accumulation buffer
    weight_error_acc: Array<Matrix>
}

impl<const L: usize> Network<L> {
    pub fn new(form: [usize; L]) -> Self {
        if form.len() <= 2 {
            panic!()
        }

        let meta = Meta::new(&form);

        Self { 
            
            biases: meta.zero_array(|l| 1..l, |i, f| (f[i], 1)),
            // {
                //     (1..form.len())
                //         .map(|i| Matrix::from_zeros((form[i], 1)))
                //         .collect()
                // },
                
            weights: meta.rand_array(|l| 0..l-1, |i, f| (f[i+1], f[i])),
            // {
            //     (0..form.len()-1)
            //     .map(|i| Matrix::rand((form[i+1], form[i])))
            //     .collect()
            // },
            
            steps: Array::new(),
            
            sums: Array::new(),
            
            errors: Array::new(),
            
            weight_errors: Array::new(),
            
            error_acc: meta.zero_array(|l| 1..l, |i, f| (f[i], 1)),
            // {
            //     (1..form.len())
            //     .map(|i| Matrix::from_zeros((form[i], 1)))
            //     .collect()
            // },


            weight_error_acc: meta.zero_array(|l| 0..l-1, |i, f| (f[i+1], f[i])),
            // {
            //     (0..form.len()-1)
            //         .map(|i| Matrix::from_zeros((form[i+1], form[i])))
            //         .collect()
            // },

            meta_data: meta,
        }
    }
    
    /// Clears propagation buffers
    pub fn clear_propagation_data(&mut self) {
        self.steps.buf.clear();
        self.sums.buf.clear();
        self.errors.buf.clear();
        self.weight_errors.buf.clear();
    }
    
    /// Clears accumulation buffers
    pub fn clear_accumulation_data(&mut self) {
        for err in self.error_acc.buf.iter_mut() {
            err.fill_eq(0.0);
        }
        for weight_err in  self.weight_error_acc.buf.iter_mut() {
            weight_err.fill_eq(0.0);
        }
    }

    /// Applies step function to one number
    fn activate(&self, n: N) -> N {
        self.meta_data.step.value(n)
    }

    /// Applies step derivative to one number
    fn d_activate(&self, n: N) -> N {
        self.meta_data.step.deriv(n)
    }

    /// Computes layer cost matrix
    fn cost_matrix(&self, exp: Matrix,  layer: usize) -> Matrix {
        Matrix::from_map(exp.shape(), |r_c| {
            self.meta_data.cost.value(exp[r_c] - self.steps[layer][r_c])
        })
    }

    /// Computes layer cost derivative matrix
    fn d_cost_matrix(&self, exp: &Matrix,  layer: usize) -> Matrix {
        Matrix::from_map(exp.shape(), |r_c| {
            self.meta_data.cost.deriv(exp[r_c] - self.steps[layer][r_c])
        })
    }

    pub fn apply_gradient(&mut self, sample_size: usize) {
        // coefficient of learn rate
        let learn_rate = self.meta_data.learn_rate / sample_size as N;

        // apply stochastic error gradient 
        for j in 0..L-1 {
            self.biases[j].add_eq(&self.error_acc[j].scale(learn_rate));
            self.weights[j].add_eq(&self.weight_error_acc[j].scale(learn_rate));
        }
    }

    pub fn forward_prop(&self, input: &Matrix) -> Matrix {
        if input.row() != self.meta_data.form[0] {
            panic!()
        }

        self.weights.buf
            .iter()
            .zip(self.biases.buf.iter())
            .fold(input.to_matrix(), |acc, (weight, bias)| {
                weight
                    .mul::<_, Matrix>(&acc)
                    .add(bias)
                    .map(|n| self.activate(n))
            })
    }

    /// Trains network against a provided set of inputs and expected outputs,  
    /// storing the error results in the 'errors' and 'weight_errors' buffers
    pub fn back_prop(&mut self, input: &Matrix, expected: &Matrix) {
        // clear previous training data
        self.clear_propagation_data();
        
        // push initial input as layer_0
        self.steps.buf.push(input.to_matrix());

        for i in 0..L-1 {
            // sum_l = weights_l * steps_l-1 + biases_l
            let sum_l = self.weights[i]
                .mul::<_, Matrix>(&self.steps[i])
                .add(&self.biases[i]);
            // step_l = activate( sum_l )
            let step_l = sum_l
                .map(|n| self.activate(n));

            self.steps.buf.push(step_l);
            self.sums.buf.push(sum_l);
        }

        // error_L = d_step( sum_L-1 ) * cost( steps_L )
        let error = self.sums[L-2]
            .map(|n| self.d_activate(n))
            .to_diagonal()
            .mul(&self.d_cost_matrix(expected, L-1));

        self.errors.buf.push(error);

        for (i, l) in (0..L-1).map(|i| (i, L-2-i)) {
            // weight_error_L = error_L * transpose( steps_L-1 )
            let weight_error_l = self.errors[L-2-l]
                .mul(&self.steps[l].transpose());

            self.weight_errors.buf.push(weight_error_l);

            if l == 0 {
                break
            }

            // error_L-1 = diagonal . d_step( sum_L-1 ) * transpose( weights_L ) * error_L
            let error_l = self.sums[l-1]
                .map(|n| self.d_activate(n))
                .to_diagonal()
                .mul::<_, Matrix>(&self.weights[l].transpose())
                .mul(&self.errors[i]);

            self.errors.buf.push(error_l);
        }

        // reverses layer error orders to ascending
        self.weight_errors.buf.reverse();
        self.errors.buf.reverse();
    }

    pub fn train(&mut self, inputs: &[Matrix], expected: &[Matrix], take: usize) {
        if inputs.len() != expected.len() {
            panic!()
        }

        self.clear_accumulation_data();

        for (i, (input, expected)) in inputs.iter().take(take).zip(expected.iter()).enumerate() {
            self.back_prop(input, expected);
            
            // accumulate errors
            for j in 0..self.errors.len() {
                self.error_acc[j].add_eq(&self.errors[j]);
                self.weight_error_acc[j].add_eq(&self.weight_errors[j]);
            }

            if i != 0 && i % self.meta_data.batch_size == 0 {
                self.apply_gradient(self.meta_data.batch_size);
                self.clear_accumulation_data();
            }
        }

        let remainder = inputs.len() % self.meta_data.batch_size;
        self.apply_gradient(remainder);
    }
}