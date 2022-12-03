// TODO: make LinAlg contain col --> index by stride so transpose doesn't have to reallocate buffer

use std::{ops::Range, fs::File};

use crate::array::Array;
use crate::array::IndexType::Back;

use super::step::Activation;
use super::cost::Cost;
use super::linalg::*;

use serde_derive::{Serialize, Deserialize};

/// Default learn coefficient
const LEARN_RATE: f32 = 0.01;
/// Default batch sample size
const BATCH_SIZE: usize = 32;
/// Default network activation function
const ACTIVATION: Activation = Activation::Tanh;
/// Default network cost function
const COST: Cost = Cost::Quad;


#[derive(Clone, Serialize, Deserialize)]
pub struct HyperData<const L: usize> {    
    // layer sizes
    form: Vec<usize>,

    // size of batch sampling
    batch_size: usize,

    // learning coefficient
    learn_rate: f32,

    // step function
    act: Activation,

    // cost function
    cost: Cost,

    // serialization directory
    dir: String,

    // controls printing of epochs
    stat_epoch: bool,

    // controls printing of error
    stat_error: bool
}

impl<const L: usize> From<[usize; L]> for HyperData<L> {
    fn from(form: [usize; L]) -> Self {
        Self { 
            form: form.to_vec(),
            batch_size: BATCH_SIZE, 
            learn_rate: LEARN_RATE, 
            act: ACTIVATION, 
            cost: COST,
            dir: String::new(),
            stat_error: false,
            stat_epoch: false
        }    
    }
}

impl<const L: usize> HyperData<L> {
    pub fn with_batch_size(&mut self, size: usize) -> &mut Self {
        self.batch_size = size;
        self
    }

    pub fn with_learn_rate(&mut self, rate: f32) -> &mut Self {
        self.learn_rate = rate;
        self
    }

    pub fn with_act(&mut self, act: Activation) -> &mut Self {
        self.act = act;
        self
    }

    pub fn with_cost(&mut self, cost: Cost) -> &mut Self {
        self.cost = cost;
        self
    }

    pub fn with_dir(&mut self, dir: &str) -> &mut Self {
        self.dir = dir.to_string();
        self
    }

    pub fn with_epoch_stats(&mut self, state: bool) -> &mut Self {
        self.stat_epoch = state;
        self
    }

    pub fn with_error_stats(&mut self, state: bool) -> &mut Self {
        self.stat_error = state;
        self
    }

    pub fn build(&self) -> Net<L> {
        Net::from_parts(self.clone())
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
    
    fn act(&self, n: f32) -> f32 {
        self.act.value(n)
    }

    fn d_act(&self, n: f32) -> f32 {
        self.act.deriv(n)
    }

    fn cost(&self, diff: f32) -> f32 {
        self.cost.value(diff)
    }

    fn d_cost(&self, diff: f32) -> f32 {
        self.cost.deriv(diff)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Net<const L: usize> {
    
    weights: Array<Matrix>,
    
    biases: Array<Vector>,
    
    /// Training Data ///

    // activations buffer
    acts: Array<Vector>,
    
    // layer summations buffer
    sums: Array<Vector>,
    
    // layer errors buffer
    err: Array<Vector>,
    
    // layer errors accumulator
    acc_err: Array<Vector>,
    
    // weight errors buffer
    w_err: Array<Matrix>,
    
    // weight error accumulator
    acc_w_err: Array<Matrix>,

    // current number of error samples
    acc_samples: usize,

    // hyper parameters
    pub data: HyperData<L>
}

impl<const L: usize> From<[usize; L]> for Net<L> {  
    fn from(form: [usize; L]) -> Self {
        Self::from_parts(HyperData::from(form))
    }
}

impl<const L: usize> Net<L> {
    pub fn from_parts(data: HyperData<L>) -> Self {
        if data.form.len() <= 2 {
            panic!()
        }

        Self { 
            acts:      data.zero_array(|l| 0..l,   |i, f| f[i]),
            sums:      data.zero_array(|l| 1..l,   |i, f| f[i]),
            err:       data.zero_array(|l| 1..l,   |i, f| f[i]),
            acc_err:   data.zero_array(|l| 1..l,   |i, f| f[i]),
            biases:    data.zero_array(|l| 1..l,   |i, f| f[i]),          
            weights:   data.rand_array(|l| 0..l-1, |i, f| (f[i+1], f[i])),
            w_err:     data.zero_array(|l| 0..l-1, |i, f| (f[i+1], f[i])),
            acc_w_err: data.zero_array(|l| 0..l-1, |i, f| (f[i+1], f[i])),
            acc_samples: 0,
            data,
        }
    }

    pub fn new(form: [usize; L]) -> HyperData<L> {
       HyperData::from(form)
    }

    pub fn save(&self) {
        let net = serde_json::to_string(&self)
            .expect("could not convert model to string!");

        let work_dir = std::env::current_dir()
            .expect("invalid working directory!");

        // create reference to absolute path
        let abs_path = work_dir.join(&self.data.dir);

        // create save file if it doesn't exist
        File::create(&abs_path)
            .expect("couldn't create model save file!");

        // write Network contents to file
        std::fs::write(&abs_path, net)
            .expect("couldn't write model to file!");
    }

    pub fn from_file(path: &str) -> Self {
        let work_dir = std::env::current_dir()
            .expect("invalid working directory!");

        // create reference to absolute path
        let abs_path = work_dir.join(&path);

        let net = std::fs::read_to_string(&abs_path)
            .expect("couldn't read model file!");
        
        serde_json::from_str(&net)
            .expect("couldn't convert from model string!")
    }

    pub fn stats(&self) -> &HyperData<L> {
        &self.data
    }
    
    pub fn clear_propagation_data(&mut self) {
        self.acts.zero();
        self.sums.zero();
        self.err.zero();
        self.w_err.zero();
    }
    
    pub fn clear_accumulation_data(&mut self) {
        self.acc_err.zero();
        self.acc_w_err.zero();

        self.acc_samples = 0;
    }

    pub fn accumulate_error(&mut self) {
        for i in 0..self.err.len() {
            self.acc_err[i].add_eq(&self.err[i]);
            self.acc_w_err[i].add_eq(&self.w_err[i]);
        }

        self.acc_samples += 1;
    }

    pub fn apply_gradient(&mut self, sample_size: usize) {
        // coefficient of learn rate
        let learn_rate = self.data.learn_rate / sample_size as f32;

        // apply stochastic error gradient 
        for j in 0..L-1 {
            self.biases[j].add_eq(&self.acc_err[j].scale(learn_rate));
            self.weights[j].add_eq(&self.acc_w_err[j].scale(learn_rate));
        }
    }

    pub fn forward_prop(&mut self, input: &Vector) -> &Vector {
        if input.row() != self.data.form[0] {
            panic!("expected data with {} rows, found shape {:?}, !", L, input.shape())
        }

        self.acts[0] = input.clone();

        for l in 0..L-1 {                  
            self.weights[l].mul_to(&self.acts[l], &mut self.sums[l]);
            self.sums[l].add_eq(&self.biases[l]);      
            self.acts[l+1] = self.sums[l].map(|n| self.data.act(n));
        }

        &self.acts[Back(0)]
    }

    pub fn back_prop(&mut self, input: &Vector, target: &Vector) {        
        // clear previous training data
        self.clear_propagation_data();
        
        // propagate and store input
        self.forward_prop(input);

        // error_L = cost' ( y - a_L ) . sum_L
        self.err[Back(0)] = target.sub(&self.acts[Back(0)]);
        self.err[Back(0)].map_eq(|n| self.data.d_cost(n));
        self.err[Back(0)].dot_eq(&self.sums[Back(0)].map(|n| self.data.d_act(n)));

        for l in 0..L-1 {
            // weight_l = error_l x activations_l-1 ^ T
            self.err[Back(l)].mul_t2_to(&self.acts[Back(1+l)], &mut self.w_err[Back(l)]);

            if l == L-2 {
                break
            }

            let (err, prev_err) = self.err.indices_mut(Back(l), Back(l+1));

            // error_l = weight_l+1 ^ T x err_l+1 . step' ( sum_l )
            self.weights[Back(l)].mul_t1_to(err, prev_err);
            prev_err.dot_eq(&self.sums[Back(l+1)].map(|n| self.data.d_act(n)));
        }
    }

    pub fn train(&mut self, inputs: &[Vector], targets: &[Vector], epochs: usize) {
        if inputs.len() != targets.len() {
            panic!("unequal amounts of input ({}) and output ({}) data!", inputs.len(), targets.len())
        }

        for epoch in 0..epochs {
            if self.data.stat_epoch {
                println!("epoch {} of {}", epoch+1, epochs);
            }

            self.clear_accumulation_data();
    
            for i in 0..inputs.len() {
                self.back_prop(&inputs[i], &targets[i]);
                self.accumulate_error();
    
                if self.acc_samples == self.data.batch_size {
                    self.apply_gradient(self.data.batch_size);
                    self.clear_accumulation_data();
                }
            }
    
            if self.acc_samples != 0 {
                self.apply_gradient(self.acc_samples);
                self.clear_accumulation_data();
            }


            if self.data.stat_error {
                let accuracy = self.accuracy(inputs, targets);
                println!("accuracy of {}", accuracy);
            }
        }
    }

    pub fn accuracy(&mut self, inputs: &[Vector], outs: &[Vector]) -> f32 {
        if inputs.len() != outs.len() {
            panic!("unequal amounts of input ({}) and output ({}) data!", inputs.len(), outs.len())
        }

        let mut correct = 0;

        for i in 0..inputs.len() {
            let out = self.forward_prop(&inputs[i]);
            if out.hot() == outs[i].hot() {
                correct += 1;
            }
        }

        correct as f32 / inputs.len() as f32
    }
}